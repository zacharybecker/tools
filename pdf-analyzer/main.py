from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from contextlib import asynccontextmanager
from typing import Optional, List, Dict
import httpx
import base64
import time
import logging

from config import get_settings, Settings
from converter import PDFConverter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConvertRequest(BaseModel):
    """Request model for PDF conversion."""
    pdf_data: str = Field(..., description="Base64-encoded PDF file")
    options: Optional[Dict] = Field(default_factory=dict)

    @validator('pdf_data')
    def validate_base64(cls, v):
        """Validate that pdf_data is valid base64."""
        try:
            decoded = base64.b64decode(v, validate=True)
            # Check if it's a reasonable size (will be enforced later)
            if len(decoded) == 0:
                raise ValueError("PDF data is empty")
            return v
        except Exception as e:
            raise ValueError(f"Invalid base64 encoding: {str(e)}")


class ConvertResponse(BaseModel):
    """Response model for PDF conversion."""
    markdown: str
    total_pages: int
    successful_pages: int
    failed_pages: int
    errors: Optional[List[Dict]] = None
    processing_time_seconds: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    litellm_configured: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    settings = get_settings()
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"LiteLLM URL: {settings.litellm_base_url}")
    logger.info(f"LiteLLM Model: {settings.litellm_model}")

    # Initialize shared HTTP client
    app.state.http_client = httpx.AsyncClient(timeout=120.0)
    app.state.settings = settings

    yield

    # Cleanup
    logger.info("Shutting down application")
    await app.state.http_client.aclose()


app = FastAPI(
    title="PDF to Markdown Converter",
    description="Convert PDF files to Markdown using LiteLLM vision models",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check(settings: Settings = Depends(get_settings)):
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        litellm_configured=bool(settings.litellm_base_url and settings.litellm_model)
    )


@app.post("/convert", response_model=ConvertResponse)
async def convert_pdf_to_markdown(
    request: ConvertRequest,
    settings: Settings = Depends(get_settings)
):
    """
    Convert a PDF file to Markdown.

    Accepts base64-encoded PDF data and returns structured Markdown.
    Processes pages in parallel using LiteLLM vision models.
    """
    start_time = time.time()

    try:
        # Decode PDF
        pdf_bytes = base64.b64decode(request.pdf_data)

        # Check size limit
        pdf_size_mb = len(pdf_bytes) / (1024 * 1024)
        if pdf_size_mb > settings.max_pdf_size_mb:
            raise HTTPException(
                status_code=400,
                detail=f"PDF size ({pdf_size_mb:.2f}MB) exceeds maximum allowed size ({settings.max_pdf_size_mb}MB)"
            )

        logger.info(f"Processing PDF of size {pdf_size_mb:.2f}MB")

        # Create converter instance for this request
        converter = PDFConverter(
            http_client=app.state.http_client,
            settings=settings
        )

        # Convert PDF to Markdown
        result = await converter.convert_pdf(pdf_bytes)

        # Add processing time
        result["processing_time_seconds"] = round(time.time() - start_time, 2)

        logger.info(
            f"Conversion completed: {result['successful_pages']}/{result['total_pages']} pages successful "
            f"in {result['processing_time_seconds']}s"
        )

        return ConvertResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Conversion failed")
        raise HTTPException(
            status_code=500,
            detail=f"Conversion failed: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "PDF to Markdown Converter",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "convert": "/convert (POST)",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
