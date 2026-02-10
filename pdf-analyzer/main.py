import os
import base64
import asyncio
import logging
from typing import Optional
from contextlib import asynccontextmanager

import fitz
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")
LITELLM_MODEL = os.getenv("LITELLM_MODEL")
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_LLM_CALLS", "5"))
MAX_PDF_SIZE_MB = int(os.getenv("MAX_PDF_SIZE_MB", "50"))


class ConvertRequest(BaseModel):
    pdf_data: str

    @validator('pdf_data')
    def validate_base64(cls, v):
        try:
            base64.b64decode(v, validate=True)
            return v
        except:
            raise ValueError("Invalid base64")


class ConvertResponse(BaseModel):
    markdown: str
    total_pages: int
    successful_pages: int
    failed_pages: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(timeout=120.0)
    app.state.semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    yield
    await app.state.client.aclose()


app = FastAPI(lifespan=lifespan)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
async def call_litellm(client, semaphore, image_bytes, page_num):
    async with semaphore:
        image_base64 = base64.b64encode(image_bytes).decode()
        response = await client.post(
            f"{LITELLM_BASE_URL}/chat/completions",
            json={
                "model": LITELLM_MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Convert this PDF page to markdown. Preserve structure and formatting."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]
                }],
                "max_tokens": 4096
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


async def convert_page(client, semaphore, image_bytes, page_num):
    try:
        markdown = await call_litellm(client, semaphore, image_bytes, page_num)
        return {"success": True, "page": page_num, "markdown": markdown}
    except Exception as e:
        logger.error(f"Page {page_num + 1} failed: {e}")
        return {"success": False, "page": page_num, "error": str(e)}


@app.post("/convert", response_model=ConvertResponse)
async def convert_pdf(request: ConvertRequest):
    pdf_bytes = base64.b64decode(request.pdf_data)

    if len(pdf_bytes) / (1024 * 1024) > MAX_PDF_SIZE_MB:
        raise HTTPException(400, f"PDF exceeds {MAX_PDF_SIZE_MB}MB")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        images.append(pix.tobytes("png"))
    doc.close()

    tasks = [convert_page(app.state.client, app.state.semaphore, img, i) for i, img in enumerate(images)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = [r for r in results if isinstance(r, dict) and r.get("success")]
    failed = [r for r in results if isinstance(r, dict) and not r.get("success")]

    markdown_parts = [f"# Page {r['page'] + 1}\n\n{r['markdown']}" for r in successful]
    combined = "\n\n---\n\n".join(markdown_parts) if markdown_parts else "# Conversion Failed"

    return ConvertResponse(
        markdown=combined,
        total_pages=len(images),
        successful_pages=len(successful),
        failed_pages=len(failed)
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
