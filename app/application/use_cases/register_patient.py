"""
Register Patient Use Case — Creates and persists a new patient record.

This use case validates the input, builds a domain entity, and delegates
persistence to the repository.  It also indexes the patient in the
search service for future queries.
"""

from __future__ import annotations

import logging
from uuid import uuid4

from app.domain.entities.patient import Gender, Patient
from app.domain.repositories.patient_repository import PatientRepository
from app.domain.services.search_service import ClinicalSearchService
from app.application.dto.patient_dto import PatientCreate

logger = logging.getLogger(__name__)


class RegisterPatientUseCase:
    """Orchestrates patient registration across relational and search stores."""

    def __init__(
        self,
        patient_repo: PatientRepository,
        search_service: ClinicalSearchService,
    ) -> None:
        self._patient_repo = patient_repo
        self._search_service = search_service

    async def execute(self, data: PatientCreate) -> Patient:
        """
        Register a new patient.

        1. Map the DTO to a domain entity.
        2. Persist in PostgreSQL via the patient repository.
        3. Index the patient record in cognitive search.

        Args:
            data: Validated patient creation DTO.

        Returns:
            The persisted ``Patient`` entity with generated fields.
        """
        patient = Patient(
            id=uuid4(),
            medical_record_number=data.medical_record_number,
            first_name=data.first_name,
            last_name=data.last_name,
            date_of_birth=data.date_of_birth,
            gender=Gender(data.gender),
            allergies=data.allergies,
            pre_existing_conditions=data.pre_existing_conditions,
        )

        persisted = await self._patient_repo.create(patient)

        # Fire-and-forget search indexing (non-critical path)
        try:
            await self._search_service.index_document(
                document_id=str(persisted.id),
                content=f"{persisted.full_name} MRN:{persisted.medical_record_number}",
                metadata={
                    "patient_id": str(persisted.id),
                    "document_type": "patient",
                },
            )
        except Exception as exc:
            logger.warning("Search indexing failed for patient %s: %s", persisted.id, exc)

        logger.info("Patient registered: %s (id=%s)", persisted.full_name, persisted.id)
        return persisted
