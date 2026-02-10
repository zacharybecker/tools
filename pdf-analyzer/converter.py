import fitz  # PyMuPDF
import asyncio
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
import httpx

from config import Settings
from litellm_client import LiteLLMClient

logger = logging.getLogger(__name__)


@dataclass
class PageResult:
    """Result of processing a single page."""
    page_num: int
    success: bool
    markdown: Optional[str] = None
    error: Optional[str] = None


class PDFConverter:
    """
    Converts PDF files to Markdown using LiteLLM vision models.

    Processes pages in parallel with concurrency control.
    """

    def __init__(self, http_client: httpx.AsyncClient, settings: Settings):
        self.http_client = http_client
        self.settings = settings
        self.semaphore = asyncio.Semaphore(settings.max_concurrent_llm_calls)
        self.litellm_client = LiteLLMClient(http_client, settings)

    def pdf_to_images(self, pdf_bytes: bytes) -> List[bytes]:
        """
        Convert PDF pages to PNG images.

        Args:
            pdf_bytes: PDF file bytes

        Returns:
            List of PNG image bytes, one per page

        Raises:
            Exception if PDF cannot be parsed
        """
        try:
            # Open PDF from bytes
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            images = []

            logger.info(f"Converting {doc.page_count} pages to images")

            for page_num in range(doc.page_count):
                page = doc[page_num]

                # Render at specified DPI (default 300)
                zoom = self.settings.image_dpi / 72
                matrix = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=matrix)

                # Convert to PNG bytes
                img_bytes = pix.tobytes("png")
                images.append(img_bytes)

                logger.debug(f"Converted page {page_num + 1} to image ({len(img_bytes)} bytes)")

            doc.close()
            logger.info(f"Successfully converted {len(images)} pages to images")

            return images

        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {str(e)}")
            raise ValueError(f"Invalid or corrupted PDF: {str(e)}")

    async def convert_page_to_markdown(
        self,
        page_image: bytes,
        page_num: int
    ) -> PageResult:
        """
        Convert a single page image to markdown.

        Uses semaphore to control concurrency.

        Args:
            page_image: PNG image bytes
            page_num: Page number (0-indexed)

        Returns:
            PageResult with success status and markdown or error
        """
        async with self.semaphore:
            try:
                logger.debug(f"Processing page {page_num + 1}")

                markdown = await self.litellm_client.convert_image_to_markdown(
                    page_image,
                    page_num
                )

                return PageResult(
                    page_num=page_num,
                    success=True,
                    markdown=markdown
                )

            except Exception as e:
                logger.error(f"Failed to convert page {page_num + 1}: {str(e)}")
                return PageResult(
                    page_num=page_num,
                    success=False,
                    error=str(e)
                )

    def combine_results(self, results: List[PageResult]) -> Dict:
        """
        Combine page results into final response.

        Args:
            results: List of PageResult objects

        Returns:
            Dictionary with markdown, stats, and errors
        """
        # Sort by page number to maintain order
        results = sorted(results, key=lambda r: r.page_num)

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        # Combine successful pages
        markdown_parts = []
        for result in successful:
            # Add page header
            markdown_parts.append(f"# Page {result.page_num + 1}\n\n{result.markdown}")

        combined_markdown = "\n\n---\n\n".join(markdown_parts)

        # If all pages failed, provide a more informative message
        if not successful:
            combined_markdown = "# Conversion Failed\n\nNo pages could be successfully converted."

        return {
            "markdown": combined_markdown,
            "total_pages": len(results),
            "successful_pages": len(successful),
            "failed_pages": len(failed),
            "errors": [
                {"page": r.page_num + 1, "error": r.error}
                for r in failed
            ] if failed else None
        }

    async def convert_pdf(self, pdf_bytes: bytes) -> Dict:
        """
        Main conversion function - orchestrates the entire process.

        Args:
            pdf_bytes: PDF file bytes

        Returns:
            Dictionary with markdown content and metadata
        """
        try:
            # Step 1: Convert PDF to images (synchronous)
            images = self.pdf_to_images(pdf_bytes)

            if not images:
                raise ValueError("PDF has no pages")

            logger.info(f"Starting parallel conversion of {len(images)} pages")

            # Step 2: Process all pages in parallel
            tasks = [
                self.convert_page_to_markdown(img, i)
                for i, img in enumerate(images)
            ]

            # Use gather with return_exceptions=True to handle failures gracefully
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Convert any exceptions to PageResult objects
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append(PageResult(
                        page_num=i,
                        success=False,
                        error=str(result)
                    ))
                else:
                    processed_results.append(result)

            # Step 3: Combine results
            final_result = self.combine_results(processed_results)

            logger.info(
                f"Conversion complete: {final_result['successful_pages']}/{final_result['total_pages']} successful"
            )

            return final_result

        except Exception as e:
            logger.exception("Fatal error during PDF conversion")
            raise
