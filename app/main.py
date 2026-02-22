"""
Clinical AI Orchestrator — FastAPI Application Entrypoint.

Configures the ASGI application with:
- OpenAPI metadata for Swagger UI and ReDoc
- CORS middleware
- Lifespan events for startup/shutdown resource management
- Versioned API routing
- Centralized error handling
"""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.middleware import register_exception_handlers, register_middleware
from app.api.v1.router import api_v1_router
from app.core.config import get_settings
from app.infrastructure.database.cosmosdb.client import CosmosDBClient

# ── Logging setup ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Handles resource initialization on startup and cleanup on shutdown.
    """
    settings = get_settings()
    logger.info("🚀 Starting %s (env=%s, mode=%s)...", settings.app_name, settings.app_env.value, settings.service_mode.value)

    # Startup: verify database connectivity
    cosmos_ok = await CosmosDBClient.health_check()
    logger.info("Cosmos DB connection: %s", "✅ OK" if cosmos_ok else "❌ FAILED")

    yield

    # Shutdown: close connections
    await CosmosDBClient.close()
    logger.info("🛑 Application shutdown complete.")


def create_app() -> FastAPI:
    """Application factory — builds and configures the FastAPI instance."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description=(
            "AI-powered Clinical Triage Engine that integrates Computer Vision "
            "and Large Language Model reasoning for intelligent medical image "
            "analysis and patient prioritization.\n\n"
            "## Architecture\n"
            "- **Hexagonal Architecture** with Domain, Application, Infrastructure, and API layers\n"
            "- **PostgreSQL** for relational patient data (SQLAlchemy 2.0)\n"
            "- **Cosmos DB / MongoDB** for semi-structured triage reports\n"
            "- **Azure Cognitive Services** for Computer Vision analysis\n"
            "- **Azure OpenAI + LangChain** for clinical reasoning\n"
            "- **Azure Cognitive Search** for full-text clinical search\n"
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Middleware ───────────────────────────────────────────────
    register_middleware(app)
    register_exception_handlers(app)

    # ── Routes ──────────────────────────────────────────────────
    app.include_router(api_v1_router, prefix=settings.api_v1_prefix)

    # ── Health check ────────────────────────────────────────────
    @app.get("/health", tags=["Health"], summary="Health check")
    async def health_check() -> dict:
        """Returns application health status and connected service info."""
        cosmos_ok = await CosmosDBClient.health_check()
        return {
            "status": "healthy",
            "service": settings.app_name,
            "version": "0.1.0",
            "environment": settings.app_env.value,
            "service_mode": settings.service_mode.value,
            "databases": {
                "postgresql": "configured",
                "cosmosdb": "connected" if cosmos_ok else "disconnected",
            },
        }

    return app


# ── Application instance ────────────────────────────────────────
app = create_app()
