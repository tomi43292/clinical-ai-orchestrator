"""
Generate Triage Report Use Case — Full AI pipeline orchestration.

This is the core use case that ties together:
1. Patient data retrieval (PostgreSQL).
2. Image analysis results collection.
3. Clinical notes retrieval.
4. LLM-based triage reasoning (LangChain).
5. Report storage (Cosmos DB).
6. Report indexing (Azure Cognitive Search).
"""

from __future__ import annotations

import logging
from uuid import UUID

from app.domain.entities.triage_report import TriageReport
from app.domain.exceptions import InsufficientDataError, PatientNotFoundError
from app.domain.repositories.patient_repository import PatientRepository
from app.domain.repositories.report_repository import TriageReportRepository
from app.domain.services.llm_service import ClinicalLLMService
from app.domain.services.search_service import ClinicalSearchService
from app.infrastructure.database.postgresql.repositories import (
    PostgresClinicalNoteRepository,
    PostgresMedicalImageRepository,
)

logger = logging.getLogger(__name__)


class GenerateTriageUseCase:
    """
    Orchestrates the end-to-end AI triage pipeline.

    Gathers all relevant patient data, invokes the LLM service for
    clinical reasoning, and persists the resulting report in the
    document store.
    """

    def __init__(
        self,
        patient_repo: PatientRepository,
        image_repo: PostgresMedicalImageRepository,
        note_repo: PostgresClinicalNoteRepository,
        report_repo: TriageReportRepository,
        llm_service: ClinicalLLMService,
        search_service: ClinicalSearchService,
    ) -> None:
        self._patient_repo = patient_repo
        self._image_repo = image_repo
        self._note_repo = note_repo
        self._report_repo = report_repo
        self._llm_service = llm_service
        self._search_service = search_service

    async def execute(
        self,
        patient_id: UUID,
        image_ids: list[UUID] | None = None,
        note_ids: list[UUID] | None = None,
    ) -> TriageReport:
        """
        Generate a triage report for a patient.

        Args:
            patient_id: UUID of the patient to assess.
            image_ids: Optional list of specific image IDs (all if None).
            note_ids: Optional list of specific note IDs (all if None).

        Returns:
            A fully populated ``TriageReport`` entity.

        Raises:
            PatientNotFoundError: If the patient does not exist.
            InsufficientDataError: If there are no images or notes to analyse.
        """
        # 1. Fetch patient
        patient = await self._patient_repo.get_by_id(patient_id)
        if patient is None:
            raise PatientNotFoundError(str(patient_id))

        # 2. Gather medical images and their analysis results
        images = await self._image_repo.get_by_patient_id(patient_id)
        if image_ids:
            images = [img for img in images if img.id in image_ids]

        image_analyses = [
            img.analysis_result for img in images
            if img.analysis_result is not None
        ]

        # 3. Gather clinical notes
        notes = await self._note_repo.get_by_patient_id(patient_id)
        if note_ids:
            notes = [n for n in notes if n.id in note_ids]

        # 4. Validate sufficient data
        if not image_analyses and not notes:
            raise InsufficientDataError(
                missing_items=["At least one analyzed image or clinical note is required."]
            )

        # 5. Generate triage via LLM
        logger.info(
            "Generating triage for patient '%s' with %d image(s) and %d note(s)...",
            patient.full_name,
            len(image_analyses),
            len(notes),
        )

        report = await self._llm_service.generate_triage(
            patient=patient,
            image_analyses=image_analyses,
            clinical_notes=notes,
        )

        # Attach referenced entity IDs
        report = TriageReport(
            id=report.id,
            patient_id=report.patient_id,
            priority_level=report.priority_level,
            clinical_summary=report.clinical_summary,
            ai_findings=report.ai_findings,
            recommended_actions=report.recommended_actions,
            suggested_questions=report.suggested_questions,
            image_ids=[img.id for img in images],
            note_ids=[n.id for n in notes],
            model_metadata=report.model_metadata,
            generated_at=report.generated_at,
        )

        # 6. Persist in Cosmos DB
        await self._report_repo.save(report)

        # 7. Index for search
        try:
            await self._search_service.index_document(
                document_id=str(report.id),
                content=report.clinical_summary,
                metadata={
                    "patient_id": str(patient_id),
                    "document_type": "triage_report",
                    "priority_level": report.priority_level.value,
                },
            )
        except Exception as exc:
            logger.warning("Failed to index triage report: %s", exc)

        logger.info(
            "Triage generated: priority=%s for patient '%s'",
            report.priority_level.value,
            patient.full_name,
        )

        return report
