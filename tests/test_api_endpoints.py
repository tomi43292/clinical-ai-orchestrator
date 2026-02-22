"""
Integration Tests for API Endpoints.

Uses the ASGI testing client to verify endpoint contracts,
response schemas, and error handling without external dependencies.
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest_asyncio.fixture
async def client():
    """Create a test client with a fresh app instance."""
    import os
    os.environ.setdefault("APP_ENV", "development")
    os.environ.setdefault("SERVICE_MODE", "mock")
    os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost/test")
    os.environ.setdefault("COSMOS_DB_CONNECTION_STRING", "mongodb://localhost:27017")
    os.environ.setdefault("COSMOS_DB_NAME", "test_db")

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_returns_ok(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "service_mode" in data

    @pytest.mark.asyncio
    async def test_health_check_contains_db_info(self, client):
        response = await client.get("/health")
        data = response.json()
        assert "databases" in data
        assert "postgresql" in data["databases"]
        assert "cosmosdb" in data["databases"]


class TestOpenAPIDocs:
    """Tests verifying API documentation is served correctly."""

    @pytest.mark.asyncio
    async def test_openapi_json(self, client):
        response = await client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "paths" in data
        assert data["info"]["title"] == "Clinical AI Triage Engine"

    @pytest.mark.asyncio
    async def test_swagger_ui(self, client):
        response = await client.get("/docs")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_redoc(self, client):
        response = await client.get("/redoc")
        assert response.status_code == 200
