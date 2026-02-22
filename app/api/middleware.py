"""
API Middleware — Cross-cutting request/response processing.

Provides request logging, performance timing, and centralized error
handling for all API endpoints.
"""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.domain.exceptions import (
    DomainError,
    ImageAnalysisError,
    InsufficientDataError,
    PatientNotFoundError,
    TriageGenerationError,
)

logger = logging.getLogger(__name__)

# ── Exception → HTTP status mapping ────────────────────────────
_EXCEPTION_STATUS_MAP: dict[type[DomainError], int] = {
    PatientNotFoundError: status.HTTP_404_NOT_FOUND,
    InsufficientDataError: status.HTTP_422_UNPROCESSABLE_ENTITY,
    ImageAnalysisError: status.HTTP_502_BAD_GATEWAY,
    TriageGenerationError: status.HTTP_502_BAD_GATEWAY,
}


def register_middleware(app: FastAPI) -> None:
    """Attach all middleware components to the FastAPI application."""

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        """Log every request with timing and a correlation ID."""
        request_id = str(uuid.uuid4())[:8]
        start_time = time.perf_counter()

        logger.info(
            "[%s] → %s %s",
            request_id,
            request.method,
            request.url.path,
        )

        response = await call_next(request)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "[%s] ← %d (%0.1fms)",
            request_id,
            response.status_code,
            elapsed_ms,
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        return response


def register_exception_handlers(app: FastAPI) -> None:
    """Register centralized exception handlers for domain errors."""

    @app.exception_handler(DomainError)
    async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
        """Convert domain exceptions to structured JSON error responses."""
        status_code = _EXCEPTION_STATUS_MAP.get(
            type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        logger.error("Domain error: %s (status=%d)", exc.message, status_code)
        return JSONResponse(
            status_code=status_code,
            content={
                "error": type(exc).__name__,
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all handler for unexpected errors."""
        logger.exception("Unhandled exception: %s", str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred. Please try again later.",
            },
        )
