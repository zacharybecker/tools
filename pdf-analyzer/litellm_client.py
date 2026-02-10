import httpx
import base64
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    RetryError
)
from typing import Optional
from config import Settings

logger = logging.getLogger(__name__)


class LiteLLMClient:
    """Client for interacting with LiteLLM API with retry logic."""

    def __init__(self, http_client: httpx.AsyncClient, settings: Settings):
        self.http_client = http_client
        self.settings = settings
        self.base_url = settings.litellm_base_url.rstrip('/')
        self.model = settings.litellm_model
        self.api_key = settings.litellm_api_key

    def _prepare_headers(self) -> dict:
        """Prepare headers for API request."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _call_api_with_retry(
        self,
        image_base64: str,
        page_num: int
    ) -> str:
        """
        Call LiteLLM API with retry logic.

        Uses tenacity for exponential backoff on failures.
        """
        @retry(
            retry=retry_if_exception_type((
                httpx.HTTPStatusError,
                httpx.TimeoutException,
                httpx.ConnectError
            )),
            wait=wait_exponential(
                multiplier=1,
                min=self.settings.retry_min_wait_seconds,
                max=self.settings.retry_max_wait_seconds
            ),
            stop=stop_after_attempt(self.settings.max_retries),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True
        )
        async def _make_request():
            logger.info(f"Calling LiteLLM for page {page_num + 1}")

            response = await self.http_client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Convert this PDF page to well-structured Markdown. "
                                    "Preserve all headings, lists, tables, code blocks, and formatting. "
                                    "Maintain the document hierarchy and structure. "
                                    "If there are tables, convert them to Markdown table format. "
                                    "Return only the Markdown content without any additional commentary."
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }],
                    "max_tokens": 4096,
                    "temperature": 0.1
                },
                headers=self._prepare_headers()
            )

            # Log status for debugging
            logger.debug(f"Response status for page {page_num + 1}: {response.status_code}")

            # Raise exception for bad status codes (will trigger retry)
            response.raise_for_status()

            # Parse response
            data = response.json()

            if "choices" not in data or len(data["choices"]) == 0:
                raise ValueError(f"Invalid LiteLLM response: no choices returned")

            markdown_content = data["choices"][0]["message"]["content"]
            logger.info(f"Successfully processed page {page_num + 1}")

            return markdown_content

        try:
            return await _make_request()
        except RetryError as e:
            logger.error(f"Failed to process page {page_num + 1} after {self.settings.max_retries} retries")
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing page {page_num + 1}: {str(e)}")
            raise

    async def convert_image_to_markdown(
        self,
        image_bytes: bytes,
        page_num: int
    ) -> str:
        """
        Convert a page image to Markdown using LiteLLM.

        Args:
            image_bytes: PNG image bytes
            page_num: Page number (0-indexed)

        Returns:
            Markdown string

        Raises:
            Exception on failure after retries
        """
        # Encode image to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # Call API with retry logic
        markdown = await self._call_api_with_retry(image_base64, page_num)

        return markdown
