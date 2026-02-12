import os
import base64
import asyncio
import logging
import tempfile
import subprocess
from contextlib import asynccontextmanager

import fitz
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")
LITELLM_MODEL = os.getenv("LITELLM_MODEL")
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_LLM_CALLS", "5"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_CONCURRENT_CONVERSIONS = int(os.getenv("MAX_CONCURRENT_CONVERSIONS", "2"))
libreoffice_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CONVERSIONS)
DPI = 300
MAX_DIMENSION_PX = int(os.getenv("MAX_DIMENSION_PX", "3000"))
OVERLAP_PX = 100
JPG_QUALITY = int(os.getenv("JPG_QUALITY", "95"))


def should_retry(exception):
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code in [429, 500, 502, 503, 504]
    return isinstance(exception, (httpx.TimeoutException, httpx.ConnectError))


class ConvertRequest(BaseModel):
    pdf_data: str

    @validator('pdf_data')
    def validate_base64(cls, v):
        try:
            base64.b64decode(v, validate=True)
            return v
        except:
            raise ValueError("Invalid base64")


class ConvertPptxRequest(BaseModel):
    pptx_data: str

    @validator('pptx_data')
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


@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=2, min=4, max=120),
    retry=retry_if_exception(should_retry)
)
async def call_litellm(client, semaphore, image_bytes, label):
    async with semaphore:
        image_base64 = base64.b64encode(image_bytes).decode()
        response = await client.post(
            f"{LITELLM_BASE_URL}/chat/completions",
            json={
                "model": LITELLM_MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Convert this PDF page section to markdown. Preserve structure and formatting."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                }],
                "max_tokens": 4096
            }
        )

        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", "30"))
            logger.warning(f"Rate limited on {label}, waiting {retry_after}s before retry")
            await asyncio.sleep(retry_after)

        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


async def convert_chunk(client, semaphore, image_bytes, page_num, chunk_num, total_chunks):
    try:
        label = f"page {page_num + 1}" if total_chunks == 1 else f"page {page_num + 1} chunk {chunk_num + 1}/{total_chunks}"
        markdown = await call_litellm(client, semaphore, image_bytes, label)
        return {"success": True, "page": page_num, "chunk": chunk_num, "markdown": markdown}
    except Exception as e:
        logger.error(f"Page {page_num + 1} chunk {chunk_num + 1} failed: {e}")
        return {"success": False, "page": page_num, "chunk": chunk_num, "error": str(e)}


def render_pdf_to_chunks(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = DPI / 72
    matrix = fitz.Matrix(zoom, zoom)

    chunks = []
    for page_num, page in enumerate(doc):
        width_px = page.rect.width * zoom
        height_px = page.rect.height * zoom

        if height_px <= MAX_DIMENSION_PX and width_px <= MAX_DIMENSION_PX:
            pix = page.get_pixmap(matrix=matrix)
            chunks.append({"image": pix.tobytes("jpeg", jpg_quality=JPG_QUALITY), "page": page_num, "chunk": 0, "total_chunks": 1})
        else:
            num_chunks = int((max(height_px, width_px) / MAX_DIMENSION_PX)) + 1
            chunk_height_pt = page.rect.height / num_chunks
            overlap_pt = OVERLAP_PX / zoom

            for chunk_idx in range(num_chunks):
                y0 = max(0, chunk_idx * chunk_height_pt - overlap_pt)
                y1 = min(page.rect.height, (chunk_idx + 1) * chunk_height_pt + overlap_pt)
                clip = fitz.Rect(0, y0, page.rect.width, y1)
                pix = page.get_pixmap(matrix=matrix, clip=clip)
                chunks.append({"image": pix.tobytes("jpeg", jpg_quality=JPG_QUALITY), "page": page_num, "chunk": chunk_idx, "total_chunks": num_chunks})

    doc.close()
    return chunks


async def process_chunks(chunks):
    tasks = [convert_chunk(app.state.client, app.state.semaphore, c["image"], c["page"], c["chunk"], c["total_chunks"])
             for c in chunks]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    page_markdowns = {}
    for r in results:
        if isinstance(r, dict) and r.get("success"):
            page_num = r["page"]
            if page_num not in page_markdowns:
                page_markdowns[page_num] = []
            page_markdowns[page_num].append((r["chunk"], r["markdown"]))

    total_pages = len(set(c["page"] for c in chunks))
    successful_pages = len(page_markdowns)

    markdown_parts = []
    for page_num in sorted(page_markdowns.keys()):
        chunks_md = [md for _, md in sorted(page_markdowns[page_num])]
        combined_page_md = "\n\n".join(chunks_md)
        markdown_parts.append(f"# Page {page_num + 1}\n\n{combined_page_md}")

    combined = "\n\n---\n\n".join(markdown_parts) if markdown_parts else "# Conversion Failed"

    return ConvertResponse(
        markdown=combined,
        total_pages=total_pages,
        successful_pages=successful_pages,
        failed_pages=total_pages - successful_pages
    )


def pptx_to_pdf(pptx_bytes):
    with tempfile.TemporaryDirectory() as tmpdir:
        pptx_path = os.path.join(tmpdir, "input.pptx")
        with open(pptx_path, "wb") as f:
            f.write(pptx_bytes)

        profile_dir = os.path.join(tmpdir, "profile")
        result = subprocess.run(
            ["libreoffice", "--headless", f"-env:UserInstallation=file://{profile_dir}",
             "--convert-to", "pdf", "--outdir", tmpdir, pptx_path],
            capture_output=True, timeout=120
        )
        if result.returncode != 0:
            raise HTTPException(500, f"LibreOffice conversion failed: {result.stderr.decode()}")

        pdf_path = os.path.join(tmpdir, "input.pdf")
        if not os.path.exists(pdf_path):
            raise HTTPException(500, "LibreOffice produced no PDF output")

        with open(pdf_path, "rb") as f:
            return f.read()


@app.post("/convert", response_model=ConvertResponse)
async def convert_pdf(request: ConvertRequest):
    pdf_bytes = base64.b64decode(request.pdf_data)

    if len(pdf_bytes) / (1024 * 1024) > MAX_FILE_SIZE_MB:
        raise HTTPException(400, f"PDF exceeds {MAX_FILE_SIZE_MB}MB")

    chunks = render_pdf_to_chunks(pdf_bytes)
    return await process_chunks(chunks)


@app.post("/convert-pptx", response_model=ConvertResponse)
async def convert_pptx(request: ConvertPptxRequest):
    pptx_bytes = base64.b64decode(request.pptx_data)

    if len(pptx_bytes) / (1024 * 1024) > MAX_FILE_SIZE_MB:
        raise HTTPException(400, f"PPTX exceeds {MAX_FILE_SIZE_MB}MB")

    async with libreoffice_semaphore:
        pdf_bytes = await asyncio.to_thread(pptx_to_pdf, pptx_bytes)
    chunks = render_pdf_to_chunks(pdf_bytes)
    return await process_chunks(chunks)


@app.get("/health")
async def health():
    return {"status": "ok"}
