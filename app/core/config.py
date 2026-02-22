"""
Clinical AI Orchestrator — Application Configuration.

Centralizes all environment-driven settings using Pydantic Settings.
Supports automatic .env file loading with validation and type coercion.
"""

from enum import Enum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceMode(str, Enum):
    """Determines whether to use real Azure services or local mocks."""

    MOCK = "mock"
    AZURE = "azure"


class AppEnvironment(str, Enum):
    """Application deployment environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Attributes are grouped by concern: app, PostgreSQL, Cosmos DB,
    Azure OpenAI, Azure Vision, Azure Cognitive Search, Azure ML,
    and service mode.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ─────────────────────────────────────────────
    app_env: AppEnvironment = AppEnvironment.DEVELOPMENT
    app_debug: bool = True
    app_name: str = "Clinical AI Orchestrator"
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"

    # ── PostgreSQL ──────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "clinical_user"
    postgres_password: str = "clinical_secret"
    postgres_db: str = "clinical_triage"
    database_url: str = Field(
        default="postgresql+asyncpg://clinical_user:clinical_secret@localhost:5432/clinical_triage"
    )

    # ── Cosmos DB / MongoDB ─────────────────────────────────────
    cosmos_db_connection_string: str = "mongodb://localhost:27017"
    cosmos_db_name: str = "clinical_reports"

    # ── Azure OpenAI ────────────────────────────────────────────
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-02-01"
    azure_openai_deployment_name: str = "gpt-4"

    # ── Azure Cognitive Services — Vision ───────────────────────
    azure_vision_endpoint: str = ""
    azure_vision_api_key: str = ""

    # ── Azure Cognitive Search ──────────────────────────────────
    azure_search_endpoint: str = ""
    azure_search_api_key: str = ""
    azure_search_index_name: str = "clinical-index"

    # ── Azure ML ────────────────────────────────────────────────
    azure_ml_subscription_id: str = ""
    azure_ml_resource_group: str = ""
    azure_ml_workspace_name: str = ""

    # ── Azure Key Vault ─────────────────────────────────────────
    azure_key_vault_url: str = ""

    # ── Service mode ────────────────────────────────────────────
    service_mode: ServiceMode = ServiceMode.MOCK

    @property
    def is_development(self) -> bool:
        return self.app_env == AppEnvironment.DEVELOPMENT

    @property
    def is_production(self) -> bool:
        return self.app_env == AppEnvironment.PRODUCTION


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings (singleton)."""
    return Settings()
