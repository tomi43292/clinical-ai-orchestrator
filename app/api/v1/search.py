"""
Search Endpoints — Clinical data search via Azure Cognitive Search.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.api.dependencies import get_search_service
from app.domain.services.search_service import ClinicalSearchService, SearchResult

router = APIRouter(prefix="/search", tags=["Search"])


class SearchResultResponse(BaseModel):
    """Search result item in the API response."""

    document_id: str
    content: str
    score: float
    highlights: list[str]
    metadata: dict[str, str]


class SearchResponse(BaseModel):
    """Paginated search results."""

    query: str
    total_results: int
    results: list[SearchResultResponse]


@router.get(
    "/clinical",
    response_model=SearchResponse,
    summary="Search clinical documents",
    description=(
        "Full-text search across clinical notes, triage reports, and patient "
        "records using Azure Cognitive Search (or in-memory mock in dev mode)."
    ),
)
async def search_clinical_data(
    search_service: Annotated[ClinicalSearchService, Depends(get_search_service)],
    q: str = Query(..., min_length=1, description="Search query"),
    top: int = Query(10, ge=1, le=50, description="Max results to return"),
    patient_id: str | None = Query(None, description="Filter by patient ID"),
    document_type: str | None = Query(None, description="Filter by document type"),
) -> SearchResponse:
    """Search indexed clinical documents with optional filters."""

    filters = {}
    if patient_id:
        filters["patient_id"] = patient_id
    if document_type:
        filters["document_type"] = document_type

    results = await search_service.search(
        query=q,
        top=top,
        filters=filters if filters else None,
    )

    return SearchResponse(
        query=q,
        total_results=len(results),
        results=[
            SearchResultResponse(
                document_id=r.document_id,
                content=r.content,
                score=r.score,
                highlights=r.highlights,
                metadata=r.metadata,
            )
            for r in results
        ],
    )
