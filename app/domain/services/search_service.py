"""
Clinical Search Service — Abstract interface for indexed clinical search.

Wraps the contract for Azure Cognitive Search (or equivalent) to allow
full-text and semantic search over clinical documents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SearchResult:
    """
    A single result from a clinical search query.

    Attributes:
        document_id: Identifier of the matched document.
        content: Text excerpt or full content of the match.
        score: Relevance score returned by the search engine.
        highlights: Highlighted snippets showing where the query matched.
        metadata: Additional metadata attached to the document.
    """

    document_id: str
    content: str
    score: float = 0.0
    highlights: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


class ClinicalSearchService(ABC):
    """
    Service contract for clinical data indexing and retrieval.

    Implementations interact with Azure Cognitive Search or a local
    mock to index clinical documents and perform search queries.
    """

    @abstractmethod
    async def index_document(
        self,
        document_id: str,
        content: str,
        metadata: dict[str, str] | None = None,
    ) -> bool:
        """
        Index a clinical document for later retrieval.

        Args:
            document_id: Unique identifier for the document.
            content: Full text content to index.
            metadata: Optional key-value metadata.

        Returns:
            ``True`` if indexing succeeded.
        """
        ...

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        top: int = 10,
        filters: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        """
        Search indexed clinical documents.

        Args:
            query: Natural-language or keyword query.
            top: Maximum number of results to return.
            filters: Optional key-value filters.

        Returns:
            Ranked list of search results.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify connectivity with the search backend."""
        ...
