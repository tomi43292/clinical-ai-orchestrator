"""
Cosmos DB Repository — Concrete implementation of TriageReportRepository.

Stores triage report documents in a MongoDB-compatible collection.
Leverages the document model for flexible schema evolution without
requiring migrations — ideal for AI-generated reports whose structure
may evolve as the models improve.
"""

from __future__ import annotations

from uuid import UUID

from app.domain.entities.triage_report import TriageReport
from app.domain.repositories.report_repository import TriageReportRepository
from app.infrastructure.database.cosmosdb.client import CosmosDBClient

COLLECTION_NAME = "triage_reports"


class CosmosTriageReportRepository(TriageReportRepository):
    """
    MongoDB-backed repository for ``TriageReport`` documents.

    Uses the ``triage_reports`` collection within the configured database.
    Documents are keyed by ``_id`` (string UUID) for cross-platform
    compatibility with Cosmos DB.
    """

    def __init__(self) -> None:
        db = CosmosDBClient.get_database()
        self._collection = db[COLLECTION_NAME]

    async def save(self, report: TriageReport) -> TriageReport:
        """
        Upsert a triage report document.

        Uses ``replace_one`` with ``upsert=True`` to support both
        creation and update semantics in a single operation.
        """
        document = report.to_document()
        await self._collection.replace_one(
            {"_id": document["_id"]},
            document,
            upsert=True,
        )
        return report

    async def get_by_id(self, report_id: UUID) -> TriageReport | None:
        """Retrieve a single triage report by its UUID."""
        document = await self._collection.find_one({"_id": str(report_id)})
        if document is None:
            return None
        return TriageReport.from_document(document)

    async def get_by_patient_id(self, patient_id: UUID) -> list[TriageReport]:
        """
        Retrieve all reports for a patient, most recent first.

        Uses a descending sort on ``generated_at`` so the treating
        physician sees the latest assessment at the top.
        """
        cursor = (
            self._collection
            .find({"patient_id": str(patient_id)})
            .sort("generated_at", -1)
        )
        documents = await cursor.to_list(length=100)
        return [TriageReport.from_document(doc) for doc in documents]

    async def delete_by_id(self, report_id: UUID) -> bool:
        """Delete a triage report by its UUID."""
        result = await self._collection.delete_one({"_id": str(report_id)})
        return result.deleted_count > 0
