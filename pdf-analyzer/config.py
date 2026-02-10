from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LiteLLM Configuration
    litellm_base_url: str
    litellm_model: str
    litellm_api_key: Optional[str] = None

    # Application Configuration
    app_name: str = "PDF to Markdown Converter"
    app_version: str = "1.0.0"
    max_concurrent_llm_calls: int = 5
    max_pdf_size_mb: int = 50

    # Retry Configuration
    max_retries: int = 5
    retry_min_wait_seconds: int = 2
    retry_max_wait_seconds: int = 60

    # PDF Processing
    image_dpi: int = 300
    image_format: str = "png"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
