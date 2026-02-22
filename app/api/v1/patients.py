"""
Patient Endpoints — CRUD operations for patient records.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status

from app.api.dependencies import (
    DBSession,
    get_patient_repo,
    get_search_service,
)
from app.application.dto.patient_dto import (
    PatientCreate,
    PatientListResponse,
    PatientResponse,
    PatientUpdate,
)
from app.application.use_cases.register_patient import RegisterPatientUseCase
from app.domain.exceptions import PatientNotFoundError
from app.domain.repositories.patient_repository import PatientRepository
from app.domain.services.search_service import ClinicalSearchService

router = APIRouter(prefix="/patients", tags=["Patients"])


def _entity_to_response(patient) -> PatientResponse:
    """Map a domain Patient entity to the API response schema."""
    return PatientResponse(
        id=patient.id,
        medical_record_number=patient.medical_record_number,
        first_name=patient.first_name,
        last_name=patient.last_name,
        full_name=patient.full_name,
        date_of_birth=patient.date_of_birth,
        age=patient.age,
        gender=patient.gender.value,
        allergies=patient.allergies,
        pre_existing_conditions=patient.pre_existing_conditions,
        created_at=patient.created_at,
        updated_at=patient.updated_at,
    )


@router.post(
    "",
    response_model=PatientResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new patient",
    description="Creates a new patient record with demographic and medical data.",
)
async def create_patient(
    data: PatientCreate,
    patient_repo: Annotated[PatientRepository, Depends(get_patient_repo)],
    search_service: Annotated[ClinicalSearchService, Depends(get_search_service)],
) -> PatientResponse:
    use_case = RegisterPatientUseCase(patient_repo, search_service)
    patient = await use_case.execute(data)
    return _entity_to_response(patient)


@router.get(
    "",
    response_model=PatientListResponse,
    summary="List all patients",
    description="Retrieve a paginated list of patients with optional search filter.",
)
async def list_patients(
    patient_repo: Annotated[PatientRepository, Depends(get_patient_repo)],
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(50, ge=1, le=100, description="Max records to return"),
    search: str | None = Query(None, description="Search by name or MRN"),
) -> PatientListResponse:
    patients, total = await patient_repo.list_all(skip=skip, limit=limit, search=search)
    return PatientListResponse(
        items=[_entity_to_response(p) for p in patients],
        total=total,
        skip=skip,
        limit=limit,
    )


@router.get(
    "/{patient_id}",
    response_model=PatientResponse,
    summary="Get patient by ID",
    description="Retrieve a single patient record with all details.",
)
async def get_patient(
    patient_id: UUID,
    patient_repo: Annotated[PatientRepository, Depends(get_patient_repo)],
) -> PatientResponse:
    patient = await patient_repo.get_by_id(patient_id)
    if patient is None:
        raise PatientNotFoundError(str(patient_id))
    return _entity_to_response(patient)
