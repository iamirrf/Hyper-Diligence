from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized env loading keeps deploy/runtime drift visible."""

    openai_api_key: str = ""
    database_url: str = "postgresql://hyper_diligence:hyper_diligence@localhost:5432/hyper_diligence"
    embedding_provider: str = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    chat_provider: str = "extractive"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    s3_bucket: str = ""
    s3_enabled: bool = True
    edgar_user_agent: str = "Hyper-Diligence research amirhosseinaref@outlook.com"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


class MissingConfigurationError(RuntimeError):
    """Raised when a runtime-only secret is required for an action."""


@lru_cache
def get_settings() -> Settings:
    return Settings()


def require_openai_api_key() -> str:
    settings = get_settings()
    if not settings.openai_api_key.strip():
        raise MissingConfigurationError("OPENAI_API_KEY is required for this operation")
    return settings.openai_api_key
