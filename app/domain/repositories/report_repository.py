"""
Triage Report Repository — Abstract interface for document persistence.

This repository targets the document store (Cosmos DB / MongoDB) where
semi-structured triage reports are stored.  The contract is intentionally
different from the relational repository because NoSQL patterns favour
document-level operations and flexible querying.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from app.domain.entities.triage_report import TriageReport


class TriageReportRepository(ABC):
    """
    Abstract repository for ``TriageReport`` documents in Cosmos DB.

    Implementations handle serialisation between domain entities and
    the underlying document format (BSON/JSON).
    """

    @abstractmethod
    async def save(self, report: TriageReport) -> TriageReport:
        """
        Insert or upsert a triage report document.

        Args:
            report: The triage report entity to persist.

        Returns:
            The persisted triage report.
        """
        ...

    @abstractmethod
    async def get_by_id(self, report_id: UUID) -> TriageReport | None:
        """
        Retrieve a single triage report by its unique identifier.

        Args:
            report_id: UUIDv4 of the report.

        Returns:
            The triage report if found, otherwise ``None``.
        """
        ...

    @abstractmethod
    async def get_by_patient_id(self, patient_id: UUID) -> list[TriageReport]:
        """
        Retrieve all triage reports for a given patient.

        Results are ordered by ``generated_at`` descending (most recent first).

        Args:
            patient_id: UUIDv4 of the patient.

        Returns:
            A list of triage reports.
        """
        ...

    @abstractmethod
    async def delete_by_id(self, report_id: UUID) -> bool:
        """
        Delete a triage report.

        Args:
            report_id: UUIDv4 of the report to delete.

        Returns:
            ``True`` if deleted, ``False`` if not found.
        """
        ...
