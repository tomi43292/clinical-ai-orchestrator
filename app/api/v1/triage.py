"""
Triage Endpoints — Generate and retrieve AI-powered triage reports.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status

from app.api.dependencies import (
    DBSession,
    get_image_repo,
    get_llm_service,
    get_note_repo,
    get_patient_repo,
    get_report_repo,
    get_search_service,
)
from app.application.dto.triage_dto import (
    AIFindingResponse,
    TriageReportResponse,
    TriageRequest,
)
from app.application.use_cases.generate_triage import GenerateTriageUseCase
from app.domain.repositories.patient_repository import PatientRepository
from app.domain.repositories.report_repository import TriageReportRepository
from app.domain.services.llm_service import ClinicalLLMService
from app.domain.services.search_service import ClinicalSearchService
from app.infrastructure.database.postgresql.repositories import (
    PostgresClinicalNoteRepository,
    PostgresMedicalImageRepository,
)

router = APIRouter(prefix="/patients/{patient_id}/triage", tags=["Triage"])


def _report_to_response(report) -> TriageReportResponse:
    """Map a TriageReport domain entity to the API response schema."""
    return TriageReportResponse(
        id=report.id,
        patient_id=report.patient_id,
        priority_level=report.priority_level.value,
        clinical_summary=report.clinical_summary,
        ai_findings=[
            AIFindingResponse(
                source=f.source,
                description=f.description,
                confidence=f.confidence,
                supporting_evidence=f.supporting_evidence,
            )
            for f in report.ai_findings
        ],
        recommended_actions=report.recommended_actions,
        suggested_questions=report.suggested_questions,
        model_metadata=report.model_metadata,
        generated_at=report.generated_at,
    )


@router.post(
    "",
    response_model=TriageReportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate a triage report",
    description=(
        "Runs the full AI triage pipeline: gathers patient images and notes, "
        "invokes the LLM reasoning engine via LangChain, and produces a "
        "structured triage assessment with priority classification."
    ),
)
async def generate_triage(
    patient_id: UUID,
    request: TriageRequest,
    patient_repo: Annotated[PatientRepository, Depends(get_patient_repo)],
    image_repo: Annotated[PostgresMedicalImageRepository, Depends(get_image_repo)],
    note_repo: Annotated[PostgresClinicalNoteRepository, Depends(get_note_repo)],
    report_repo: Annotated[TriageReportRepository, Depends(get_report_repo)],
    llm_service: Annotated[ClinicalLLMService, Depends(get_llm_service)],
    search_service: Annotated[ClinicalSearchService, Depends(get_search_service)],
) -> TriageReportResponse:
    """Generate an AI-powered triage assessment for a patient."""
    use_case = GenerateTriageUseCase(
        patient_repo=patient_repo,
        image_repo=image_repo,
        note_repo=note_repo,
        report_repo=report_repo,
        llm_service=llm_service,
        search_service=search_service,
    )

    report = await use_case.execute(
        patient_id=patient_id,
        image_ids=request.image_ids,
        note_ids=request.note_ids,
    )

    return _report_to_response(report)


@router.get(
    "/reports",
    response_model=list[TriageReportResponse],
    summary="List triage reports for a patient",
    description="Retrieve all triage reports for a patient, most recent first.",
)
async def list_triage_reports(
    patient_id: UUID,
    report_repo: Annotated[TriageReportRepository, Depends(get_report_repo)],
) -> list[TriageReportResponse]:
    reports = await report_repo.get_by_patient_id(patient_id)
    return [_report_to_response(r) for r in reports]
