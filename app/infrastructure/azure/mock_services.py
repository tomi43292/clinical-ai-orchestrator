"""
Mock Service Implementations — Local development stubs.

These implementations satisfy the domain service interfaces without
requiring real Azure credentials.  They return realistic synthetic
medical data that exercises the full triage pipeline, making it
possible to develop and demo the system entirely offline.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timezone
from uuid import uuid4

from app.domain.entities.clinical_note import ClinicalNote
from app.domain.entities.medical_image import (
    DetectedAnomaly,
    ImageAnalysisResult,
    ImageModality,
)
from app.domain.entities.patient import Patient
from app.domain.entities.triage_report import AIFinding, PriorityLevel, TriageReport
from app.domain.services.llm_service import ClinicalLLMService
from app.domain.services.search_service import ClinicalSearchService, SearchResult
from app.domain.services.vision_service import VisionAnalysisService

logger = logging.getLogger(__name__)

# ── Synthetic medical data pools ────────────────────────────────
_ANOMALY_TEMPLATES = [
    ("pulmonary_opacity", "Possible opacification in the right lower lobe", "right_lower_lobe"),
    ("cardiomegaly", "Cardiac silhouette appears enlarged", "mediastinum"),
    ("pleural_effusion", "Small pleural effusion noted on left side", "left_costophrenic_angle"),
    ("fracture", "Suspected cortical disruption in the 7th rib", "right_lateral_chest"),
    ("consolidation", "Dense consolidation pattern in left upper lobe", "left_upper_lobe"),
    ("no_anomaly", "No significant abnormalities detected", "global"),
]

_CLINICAL_TAGS = [
    "opacity", "effusion", "consolidation", "normal", "cardiomegaly",
    "pneumothorax", "nodule", "mass", "atelectasis", "emphysema",
]

_TRIAGE_SUMMARIES = {
    PriorityLevel.CRITICAL: (
        "Critical findings detected. The AI analysis identified significant anomalies "
        "in the diagnostic imaging that, combined with the patient's clinical presentation, "
        "suggest a condition requiring immediate medical attention. Urgent specialist "
        "consultation is strongly recommended."
    ),
    PriorityLevel.HIGH: (
        "High-priority findings identified. The correlation between imaging anomalies "
        "and clinical notes indicates a condition that should be evaluated promptly. "
        "Further diagnostic workup is recommended within 24 hours."
    ),
    PriorityLevel.MEDIUM: (
        "Moderate findings noted. The imaging shows some areas of concern that should "
        "be monitored. Clinical notes are consistent with a non-emergent but notable "
        "condition. Follow-up within 72 hours is suggested."
    ),
    PriorityLevel.LOW: (
        "Minor or no significant findings. The imaging study appears largely normal, "
        "and clinical notes describe symptoms that are self-limiting or well-controlled. "
        "Routine follow-up is adequate."
    ),
}


class MockVisionService(VisionAnalysisService):
    """
    Mock Computer Vision service returning synthetic anomaly detection results.

    Simulates Azure Cognitive Services Computer Vision by randomly selecting
    from a pool of realistic medical findings.
    """

    async def analyze_image(
        self,
        image_bytes: bytes,
        modality: ImageModality,
        body_region: str = "unknown",
    ) -> ImageAnalysisResult:
        """Return synthetic analysis results based on image size heuristics."""
        logger.info(
            "🔬 [MOCK] Analyzing %s image (%d bytes) for region '%s'...",
            modality.value,
            len(image_bytes),
            body_region,
        )

        # Simulate 1-3 detected anomalies
        num_anomalies = random.randint(0, 3)
        selected = random.sample(_ANOMALY_TEMPLATES, min(num_anomalies, len(_ANOMALY_TEMPLATES)))

        anomalies = [
            DetectedAnomaly(
                label=label,
                confidence=round(random.uniform(0.65, 0.98), 2),
                region=region,
                bounding_box=[
                    round(random.uniform(50, 200), 1),
                    round(random.uniform(50, 200), 1),
                    round(random.uniform(80, 150), 1),
                    round(random.uniform(80, 150), 1),
                ],
            )
            for label, _, region in selected
        ]

        descriptions = [desc for _, desc, _ in selected]
        description = ". ".join(descriptions) if descriptions else "No significant findings."

        return ImageAnalysisResult(
            anomalies=anomalies,
            description=description,
            raw_tags=random.sample(_CLINICAL_TAGS, min(5, len(_CLINICAL_TAGS))),
            model_version="mock-cv-1.0",
        )

    async def health_check(self) -> bool:
        return True


class MockLLMService(ClinicalLLMService):
    """
    Mock LLM triage service returning synthetic clinical assessments.

    Simulates LangChain-based reasoning by generating structured triage
    reports from the provided patient data and analysis results.
    """

    async def generate_triage(
        self,
        patient: Patient,
        image_analyses: list[ImageAnalysisResult],
        clinical_notes: list[ClinicalNote],
    ) -> TriageReport:
        """Generate a synthetic triage report combining all inputs."""
        logger.info(
            "🧠 [MOCK] Generating triage for patient '%s' with %d image(s) and %d note(s)...",
            patient.full_name,
            len(image_analyses),
            len(clinical_notes),
        )

        # Determine priority based on anomaly count and confidence
        total_anomalies = sum(len(a.anomalies) for a in image_analyses)
        max_confidence = max(
            (anomaly.confidence for a in image_analyses for anomaly in a.anomalies),
            default=0.0,
        )

        if total_anomalies >= 3 or max_confidence > 0.95:
            priority = PriorityLevel.CRITICAL
        elif total_anomalies >= 2 or max_confidence > 0.85:
            priority = PriorityLevel.HIGH
        elif total_anomalies >= 1:
            priority = PriorityLevel.MEDIUM
        else:
            priority = PriorityLevel.LOW

        # Build AI findings from image analyses
        findings = []
        for analysis in image_analyses:
            for anomaly in analysis.anomalies:
                findings.append(
                    AIFinding(
                        source="vision",
                        description=f"Detected '{anomaly.label}' in region '{anomaly.region}'",
                        confidence=anomaly.confidence,
                        supporting_evidence=analysis.description,
                    )
                )

        # Add LLM-based correlation finding
        if clinical_notes and findings:
            note_summary = "; ".join(n.content[:100] for n in clinical_notes[:3])
            findings.append(
                AIFinding(
                    source="correlation",
                    description=(
                        f"Cross-referencing {len(findings)} imaging findings with "
                        f"clinical notes: {note_summary}..."
                    ),
                    confidence=round(random.uniform(0.70, 0.95), 2),
                    supporting_evidence="LLM clinical correlation analysis",
                )
            )

        return TriageReport(
            id=uuid4(),
            patient_id=patient.id,
            priority_level=priority,
            clinical_summary=_TRIAGE_SUMMARIES[priority],
            ai_findings=findings,
            recommended_actions=[
                "Review imaging findings with radiologist",
                "Correlate with patient's medication history",
                "Consider additional diagnostic studies if clinically indicated",
                "Schedule follow-up based on priority level",
            ],
            suggested_questions=[
                "Has the patient experienced similar symptoms before?",
                "Any recent changes in medication?",
                "Family history of similar conditions?",
                "Duration and progression of current symptoms?",
            ],
            image_ids=[],
            note_ids=[],
            model_metadata={
                "vision_model": "mock-cv-1.0",
                "llm_model": "mock-gpt-4",
                "pipeline_version": "1.0.0",
            },
            generated_at=datetime.now(tz=timezone.utc),
        )

    async def health_check(self) -> bool:
        return True


class MockSearchService(ClinicalSearchService):
    """
    Mock Cognitive Search service with in-memory document storage.

    Provides basic keyword matching for local development and testing.
    """

    def __init__(self) -> None:
        self._documents: dict[str, dict] = {}

    async def index_document(
        self,
        document_id: str,
        content: str,
        metadata: dict[str, str] | None = None,
    ) -> bool:
        """Store a document in the in-memory index."""
        logger.info("🔍 [MOCK] Indexing document '%s'...", document_id)
        self._documents[document_id] = {
            "id": document_id,
            "content": content,
            **(metadata or {}),
        }
        return True

    async def search(
        self,
        query: str,
        *,
        top: int = 10,
        filters: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        """Simple keyword search across stored documents."""
        logger.info("🔍 [MOCK] Searching for '%s'...", query)
        results: list[SearchResult] = []
        query_lower = query.lower()

        for doc in self._documents.values():
            content = doc.get("content", "")
            if query_lower in content.lower():
                results.append(
                    SearchResult(
                        document_id=doc["id"],
                        content=content[:200],
                        score=round(random.uniform(0.5, 1.0), 2),
                        highlights=[content[:100]],
                        metadata={
                            k: v for k, v in doc.items()
                            if k not in ("id", "content")
                        },
                    )
                )

        return results[:top]

    async def health_check(self) -> bool:
        return True
