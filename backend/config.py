from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",
    )

    LLM_MODEL: str = "gemma2:latest"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "mevzuat_rag"
    EMBED_MODEL: str = "ytu-ce-cosmos/turkish-e5-large"

    @field_validator("QDRANT_URL")
    @classmethod
    def validate_qdrant_url(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("QDRANT_URL boş olamaz")
        return value