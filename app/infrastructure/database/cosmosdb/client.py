"""
Cosmos DB / MongoDB Client — Async document store connection.

Azure Cosmos DB exposes a MongoDB-compatible API, allowing us to use
Motor (async MongoDB driver) for both local development (plain MongoDB)
and production (Cosmos DB with MongoDB vCore or RU-based API).
"""

from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import get_settings


class CosmosDBClient:
    """
    Singleton-style async client for Cosmos DB / MongoDB.

    Attributes:
        _client: The underlying Motor async client instance.
        _database: Reference to the target database.
    """

    _client: AsyncIOMotorClient | None = None
    _database: AsyncIOMotorDatabase | None = None

    @classmethod
    def get_client(cls) -> AsyncIOMotorClient:
        """Return the shared Motor client, creating it lazily if needed."""
        if cls._client is None:
            settings = get_settings()
            cls._client = AsyncIOMotorClient(settings.cosmos_db_connection_string)
        return cls._client

    @classmethod
    def get_database(cls) -> AsyncIOMotorDatabase:
        """Return the target database handle."""
        if cls._database is None:
            settings = get_settings()
            client = cls.get_client()
            cls._database = client[settings.cosmos_db_name]
        return cls._database

    @classmethod
    async def close(cls) -> None:
        """Gracefully close the underlying connection."""
        if cls._client is not None:
            cls._client.close()
            cls._client = None
            cls._database = None

    @classmethod
    async def health_check(cls) -> bool:
        """Ping the database to verify connectivity."""
        try:
            client = cls.get_client()
            await client.admin.command("ping")
            return True
        except Exception:
            return False
