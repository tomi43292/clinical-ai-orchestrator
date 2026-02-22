"""
Clinical LLM Service — Abstract interface for LLM-based triage reasoning.

Encapsulates the orchestration of large language models (Azure OpenAI,
Hugging Face) through LangChain to combine image findings with clinical
notes and produce a structured triage assessment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.entities.clinical_note import ClinicalNote
from app.domain.entities.medical_image import ImageAnalysisResult
from app.domain.entities.patient import Patient
from app.domain.entities.triage_report import TriageReport


class ClinicalLLMService(ABC):
    """
    Service contract for AI-driven clinical triage reasoning.

    Implementations are expected to use LangChain (or equivalent) to
    run a multi-step reasoning chain that correlates vision analysis
    findings with clinical notes and patient demographics.
    """

    @abstractmethod
    async def generate_triage(
        self,
        patient: Patient,
        image_analyses: list[ImageAnalysisResult],
        clinical_notes: list[ClinicalNote],
    ) -> TriageReport:
        """
        Generate a triage report by orchestrating LLM reasoning.

        The implementation is expected to:
        1. Build context from patient demographics and medical history.
        2. Correlate image analysis results with note content.
        3. Classify urgency and produce actionable recommendations.

        Args:
            patient: The patient entity with demographics and history.
            image_analyses: CV analysis results for the patient's images.
            clinical_notes: Free-text clinical notes to factor in.

        Returns:
            A fully populated TriageReport entity.

        Raises:
            TriageGenerationError: If the LLM pipeline fails.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify the LLM backend is reachable and responding."""
        ...
