"""
Azure Cognitive Search Client — Clinical document indexing and retrieval.

Implements the ``ClinicalSearchService`` domain interface using
Azure AI Search (formerly Azure Cognitive Search) for indexing
clinical notes and triage reports, enabling RAG-style queries.
"""

from __future__ import annotations

import logging

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
)

from app.core.config import get_settings
from app.domain.exceptions import SearchIndexError
from app.domain.services.search_service import ClinicalSearchService, SearchResult

logger = logging.getLogger(__name__)


class AzureCognitiveSearchService(ClinicalSearchService):
    """
    Azure AI Search implementation of ``ClinicalSearchService``.

    Manages an index called ``clinical-index`` with searchable text
    fields and filterable metadata for clinical document retrieval.
    """

    def __init__(self) -> None:
        settings = get_settings()
        credential = AzureKeyCredential(settings.azure_search_api_key)

        self._search_client = SearchClient(
            endpoint=settings.azure_search_endpoint,
            index_name=settings.azure_search_index_name,
            credential=credential,
        )
        self._index_client = SearchIndexClient(
            endpoint=settings.azure_search_endpoint,
            credential=credential,
        )
        self._index_name = settings.azure_search_index_name

    async def ensure_index_exists(self) -> None:
        """
        Create the search index if it does not already exist.

        The index schema includes a searchable ``content`` field and
        filterable metadata fields for patient_id, note_type, etc.
        """
        try:
            fields = [
                SimpleField(
                    name="id",
                    type=SearchFieldDataType.String,
                    key=True,
                    filterable=True,
                ),
                SearchableField(
                    name="content",
                    type=SearchFieldDataType.String,
                    analyzer_name="en.lucene",
                ),
                SimpleField(
                    name="patient_id",
                    type=SearchFieldDataType.String,
                    filterable=True,
                ),
                SimpleField(
                    name="document_type",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    facetable=True,
                ),
                SearchableField(
                    name="author",
                    type=SearchFieldDataType.String,
                ),
                SimpleField(
                    name="created_at",
                    type=SearchFieldDataType.DateTimeOffset,
                    sortable=True,
                ),
            ]

            index = SearchIndex(name=self._index_name, fields=fields)
            self._index_client.create_or_update_index(index)
            logger.info("Search index '%s' ensured.", self._index_name)

        except Exception as exc:
            logger.error("Failed to create search index: %s", str(exc))
            raise SearchIndexError(reason=str(exc)) from exc

    async def index_document(
        self,
        document_id: str,
        content: str,
        metadata: dict[str, str] | None = None,
    ) -> bool:
        """Index a clinical document for full-text retrieval."""
        try:
            document = {
                "id": document_id,
                "content": content,
                **(metadata or {}),
            }
            result = self._search_client.upload_documents(documents=[document])
            success = all(r.succeeded for r in result)

            if success:
                logger.info("Document '%s' indexed successfully.", document_id)
            else:
                logger.warning("Document '%s' indexing had failures.", document_id)

            return success

        except Exception as exc:
            logger.error("Failed to index document '%s': %s", document_id, str(exc))
            raise SearchIndexError(reason=str(exc)) from exc

    async def search(
        self,
        query: str,
        *,
        top: int = 10,
        filters: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        """
        Execute a full-text search across indexed clinical documents.

        Supports OData-style filter expressions via the ``filters`` dict.
        """
        try:
            filter_expression = None
            if filters:
                clauses = [f"{key} eq '{value}'" for key, value in filters.items()]
                filter_expression = " and ".join(clauses)

            results = self._search_client.search(
                search_text=query,
                top=top,
                filter=filter_expression,
                highlight_fields="content",
            )

            search_results: list[SearchResult] = []
            for result in results:
                search_results.append(
                    SearchResult(
                        document_id=result["id"],
                        content=result.get("content", ""),
                        score=result.get("@search.score", 0.0),
                        highlights=result.get("@search.highlights", {}).get(
                            "content", []
                        ),
                        metadata={
                            "patient_id": result.get("patient_id", ""),
                            "document_type": result.get("document_type", ""),
                        },
                    )
                )

            logger.info("Search returned %d results for query: '%s'", len(search_results), query)
            return search_results

        except Exception as exc:
            logger.error("Search query failed: %s", str(exc))
            raise SearchIndexError(reason=str(exc)) from exc

    async def health_check(self) -> bool:
        """Verify connectivity with Azure Cognitive Search."""
        try:
            self._index_client.list_index_names()
            return True
        except Exception:
            return False
