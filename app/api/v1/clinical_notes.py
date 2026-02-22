"""
Clinical Notes Endpoints — Create and list clinical observations.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, status

from app.api.dependencies import (
    DBSession,
    get_note_repo,
    get_patient_repo,
    get_search_service,
)
from app.application.dto.triage_dto import ClinicalNoteCreate, ClinicalNoteResponse
from app.domain.entities.clinical_note import ClinicalNote
from app.domain.exceptions import PatientNotFoundError
from app.domain.repositories.patient_repository import PatientRepository
from app.domain.services.search_service import ClinicalSearchService
from app.infrastructure.database.postgresql.repositories import PostgresClinicalNoteRepository

router = APIRouter(prefix="/patients/{patient_id}/clinical-notes", tags=["Clinical Notes"])


@router.post(
    "",
    response_model=ClinicalNoteResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a clinical note",
    description="Record a clinical observation for a patient. Notes are indexed for search.",
)
async def create_clinical_note(
    patient_id: UUID,
    data: ClinicalNoteCreate,
    patient_repo: Annotated[PatientRepository, Depends(get_patient_repo)],
    note_repo: Annotated[PostgresClinicalNoteRepository, Depends(get_note_repo)],
    search_service: Annotated[ClinicalSearchService, Depends(get_search_service)],
) -> ClinicalNoteResponse:
    """Create a new clinical note and index it for search."""

    # Verify patient exists
    patient = await patient_repo.get_by_id(patient_id)
    if patient is None:
        raise PatientNotFoundError(str(patient_id))

    # Create domain entity
    note = ClinicalNote(
        id=uuid4(),
        patient_id=patient_id,
        author=data.author,
        content=data.content,
        note_type=data.note_type,
    )

    # Persist in PostgreSQL
    persisted = await note_repo.create(note)

    # Index in search (non-critical)
    try:
        await search_service.index_document(
            document_id=str(persisted.id),
            content=persisted.content,
            metadata={
                "patient_id": str(patient_id),
                "document_type": "clinical_note",
                "author": persisted.author,
            },
        )
    except Exception:
        pass  # Search indexing failures are non-critical

    return ClinicalNoteResponse(
        id=persisted.id,
        patient_id=persisted.patient_id,
        author=persisted.author,
        content=persisted.content,
        note_type=persisted.note_type,
        created_at=persisted.created_at,
    )


@router.get(
    "",
    response_model=list[ClinicalNoteResponse],
    summary="List clinical notes for a patient",
    description="Retrieve all clinical notes for a given patient, newest first.",
)
async def list_clinical_notes(
    patient_id: UUID,
    note_repo: Annotated[PostgresClinicalNoteRepository, Depends(get_note_repo)],
) -> list[ClinicalNoteResponse]:
    notes = await note_repo.get_by_patient_id(patient_id)
    return [
        ClinicalNoteResponse(
            id=n.id,
            patient_id=n.patient_id,
            author=n.author,
            content=n.content,
            note_type=n.note_type,
            created_at=n.created_at,
        )
        for n in notes
    ]
