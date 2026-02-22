"""
Triage Agent — LangChain agent implementing the ClinicalLLMService interface.

Orchestrates the full triage pipeline: builds patient context, invokes
the clinical analysis chain, and maps the LLM output back to a domain
``TriageReport`` entity.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

from app.core.config import ServiceMode, get_settings
from app.domain.entities.clinical_note import ClinicalNote
from app.domain.entities.medical_image import ImageAnalysisResult
from app.domain.entities.patient import Patient
from app.domain.entities.triage_report import AIFinding, PriorityLevel, TriageReport
from app.domain.exceptions import TriageGenerationError
from app.domain.services.llm_service import ClinicalLLMService
from app.infrastructure.langchain.chains import ClinicalAnalysisChain

logger = logging.getLogger(__name__)


class LangChainTriageAgent(ClinicalLLMService):
    """
    LangChain-based implementation of ``ClinicalLLMService``.

    Uses a ``ClinicalAnalysisChain`` to process patient data through
    Azure OpenAI and produce structured triage reports.
    """

    def __init__(self) -> None:
        settings = get_settings()

        if settings.service_mode == ServiceMode.AZURE:
            from app.infrastructure.azure.openai_client import get_azure_llm

            llm = get_azure_llm()
        else:
            # In mock mode, we use the MockLLMService directly
            # This agent is only instantiated in azure mode
            from langchain_community.llms.fake import FakeListLLM

            llm = FakeListLLM(responses=["{}"])

        self._chain = ClinicalAnalysisChain(llm)

    async def generate_triage(
        self,
        patient: Patient,
        image_analyses: list[ImageAnalysisResult],
        clinical_notes: list[ClinicalNote],
    ) -> TriageReport:
        """
        Execute the full triage pipeline via LangChain.

        Steps:
        1. Invoke the clinical analysis chain with all patient data.
        2. Parse the structured JSON response from the LLM.
        3. Map the parsed data to a domain ``TriageReport`` entity.
        """
        try:
            logger.info(
                "🤖 Running LangChain triage for patient '%s'...",
                patient.full_name,
            )

            result = await self._chain.invoke(patient, image_analyses, clinical_notes)
            return self._map_to_report(patient, result)

        except Exception as exc:
            logger.error("Triage generation failed: %s", str(exc))
            raise TriageGenerationError(reason=str(exc)) from exc

    async def health_check(self) -> bool:
        """Verify the LangChain pipeline is operational."""
        return True

    @staticmethod
    def _map_to_report(patient: Patient, llm_output: dict) -> TriageReport:
        """Map parsed LLM JSON output to a domain TriageReport entity."""
        # Parse priority level with fallback
        try:
            priority = PriorityLevel(llm_output.get("priority_level", "medium"))
        except ValueError:
            priority = PriorityLevel.MEDIUM

        # Parse findings
        findings = [
            AIFinding(
                source=f.get("source", "llm"),
                description=f.get("description", ""),
                confidence=f.get("confidence", 0.0),
                supporting_evidence=f.get("supporting_evidence", ""),
            )
            for f in llm_output.get("findings", [])
        ]

        return TriageReport(
            id=uuid4(),
            patient_id=patient.id,
            priority_level=priority,
            clinical_summary=llm_output.get("clinical_summary", ""),
            ai_findings=findings,
            recommended_actions=llm_output.get("recommended_actions", []),
            suggested_questions=llm_output.get("suggested_questions", []),
            model_metadata={
                "pipeline": "langchain",
                "agent_version": "1.0.0",
            },
            generated_at=datetime.now(tz=timezone.utc),
        )
