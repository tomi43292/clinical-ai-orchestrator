"""
Async SQLAlchemy Session Factory — PostgreSQL connection management.

Provides an async engine and session maker configured for the application's
PostgreSQL instance.  The ``get_db_session`` async generator is designed
to be used as a FastAPI dependency.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings

settings = get_settings()

# ── Async Engine ────────────────────────────────────────────────
engine = create_async_engine(
    settings.database_url,
    echo=settings.app_debug,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,
)

# ── Session Factory ─────────────────────────────────────────────
async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Yield an async database session.

    Intended for use as a FastAPI ``Depends`` dependency.  The session
    is automatically closed after the request completes.

    Yields:
        An ``AsyncSession`` bound to the application's PostgreSQL engine.
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
