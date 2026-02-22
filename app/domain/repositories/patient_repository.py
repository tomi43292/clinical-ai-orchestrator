"""
Patient Repository — Abstract interface for patient persistence.

Defines the contract that any concrete persistence adapter (PostgreSQL,
in-memory, etc.) must satisfy.  This keeps the domain layer decoupled
from infrastructure details — a core Hexagonal Architecture principle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from app.domain.entities.patient import Patient


class PatientRepository(ABC):
    """
    Abstract repository for ``Patient`` aggregate root.

    Concrete implementations live in the infrastructure layer and handle
    mapping between domain entities and their storage representations.
    """

    @abstractmethod
    async def create(self, patient: Patient) -> Patient:
        """
        Persist a new patient record.

        Args:
            patient: The patient entity to persist.

        Returns:
            The persisted patient with generated ``id`` and timestamps.
        """
        ...

    @abstractmethod
    async def get_by_id(self, patient_id: UUID) -> Patient | None:
        """
        Retrieve a patient by their unique identifier.

        Args:
            patient_id: UUIDv4 of the patient.

        Returns:
            The patient entity if found, otherwise ``None``.
        """
        ...

    @abstractmethod
    async def list_all(
        self,
        *,
        skip: int = 0,
        limit: int = 50,
        search: str | None = None,
    ) -> tuple[list[Patient], int]:
        """
        List patients with pagination and optional search.

        Args:
            skip: Number of records to skip (offset).
            limit: Maximum number of records to return.
            search: Optional full-text search filter on name/MRN.

        Returns:
            A tuple of ``(patients, total_count)``.
        """
        ...

    @abstractmethod
    async def update(self, patient: Patient) -> Patient:
        """
        Update an existing patient record.

        Args:
            patient: Patient entity with updated fields.

        Returns:
            The updated patient entity.
        """
        ...

    @abstractmethod
    async def delete(self, patient_id: UUID) -> bool:
        """
        Soft-delete or remove a patient record.

        Args:
            patient_id: UUIDv4 of the patient to delete.

        Returns:
            ``True`` if deleted, ``False`` if not found.
        """
        ...
