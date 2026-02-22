"""
Clinical Analysis Chain — LangChain chain for structured triage output.

Composes the clinical prompt template with an LLM and a JSON output
parser to produce consistently structured triage assessments.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.domain.entities.clinical_note import ClinicalNote
from app.domain.entities.medical_image import ImageAnalysisResult
from app.domain.entities.patient import Patient
from app.infrastructure.langchain.prompts import TRIAGE_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)


class ClinicalAnalysisChain:
    """
    LangChain chain that processes patient data through the clinical
    triage prompt and returns a parsed JSON assessment.

    Usage:
        chain = ClinicalAnalysisChain(llm)
        result = await chain.invoke(patient, image_analyses, clinical_notes)
    """

    def __init__(self, llm: Any) -> None:
        """
        Initialize with a LangChain-compatible LLM.

        Args:
            llm: A LangChain chat model (e.g., ``AzureChatOpenAI``).
        """
        self._chain = TRIAGE_ANALYSIS_PROMPT | llm | StrOutputParser()

    async def invoke(
        self,
        patient: Patient,
        image_analyses: list[ImageAnalysisResult],
        clinical_notes: list[ClinicalNote],
    ) -> dict:
        """
        Run the clinical analysis chain.

        Args:
            patient: Patient entity with demographics.
            image_analyses: List of CV analysis results.
            clinical_notes: List of clinical notes.

        Returns:
            Parsed JSON dict with triage assessment fields.
        """
        # Format imaging findings as readable text
        imaging_text = self._format_imaging_findings(image_analyses)
        notes_text = self._format_clinical_notes(clinical_notes)

        input_data = {
            "patient_name": patient.full_name,
            "patient_age": str(patient.age or "unknown"),
            "patient_gender": patient.gender.value,
            "patient_mrn": patient.medical_record_number,
            "allergies": ", ".join(patient.allergies) or "None known",
            "pre_existing_conditions": (
                ", ".join(patient.pre_existing_conditions) or "None reported"
            ),
            "imaging_findings": imaging_text,
            "clinical_notes": notes_text,
        }

        logger.info("Invoking clinical analysis chain for patient '%s'...", patient.full_name)

        raw_output = await self._chain.ainvoke(input_data)

        return self._parse_output(raw_output)

    @staticmethod
    def _format_imaging_findings(analyses: list[ImageAnalysisResult]) -> str:
        """Convert image analysis results to a human-readable summary."""
        if not analyses:
            return "No imaging studies available."

        sections = []
        for i, analysis in enumerate(analyses, 1):
            lines = [f"### Study {i}"]
            lines.append(f"**Description**: {analysis.description}")
            lines.append(f"**Tags**: {', '.join(analysis.raw_tags)}")

            if analysis.anomalies:
                lines.append("**Detected Anomalies**:")
                for anomaly in analysis.anomalies:
                    lines.append(
                        f"  - {anomaly.label} (confidence: {anomaly.confidence:.0%}) "
                        f"— Region: {anomaly.region}"
                    )
            else:
                lines.append("**No anomalies detected.**")

            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    @staticmethod
    def _format_clinical_notes(notes: list[ClinicalNote]) -> str:
        """Convert clinical notes to a chronological text summary."""
        if not notes:
            return "No clinical notes available."

        entries = []
        for note in notes:
            timestamp = note.created_at.isoformat() if note.created_at else "N/A"
            entries.append(
                f"**[{timestamp}] {note.author} ({note.note_type})**:\n{note.content}"
            )

        return "\n\n".join(entries)

    @staticmethod
    def _parse_output(raw_output: str) -> dict:
        """
        Parse the LLM output as JSON, handling markdown code fences.

        Falls back to a basic structure if parsing fails to ensure
        the pipeline never crashes on malformed LLM output.
        """
        # Strip markdown code fences if present
        cleaned = raw_output.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM output as JSON, returning raw text.")
            return {
                "priority_level": "medium",
                "clinical_summary": raw_output,
                "findings": [],
                "recommended_actions": ["Manual review required — LLM output was unstructured."],
                "suggested_questions": [],
            }
