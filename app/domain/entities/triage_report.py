"""
Triage Report Entity — AI-generated clinical assessment document.

This is the final output of the orchestration pipeline.  It combines
Computer Vision findings with LLM clinical reasoning into a structured
document that lives in the document store (Cosmos DB / MongoDB).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4


class PriorityLevel(str, Enum):
    """Clinical urgency classification."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True, slots=True)
class AIFinding:
    """
    Individual finding produced by the AI analysis pipeline.

    Attributes:
        source: Origin of the finding ("vision" | "llm" | "correlation").
        description: Human-readable description of the finding.
        confidence: Confidence score (0-1).
        supporting_evidence: Raw data that supports this finding.
    """

    source: str
    description: str
    confidence: float = 0.0
    supporting_evidence: str = ""


@dataclass(frozen=True, slots=True)
class TriageReport:
    """
    AI-generated triage report stored as a document in Cosmos DB.

    Attributes:
        id: Unique identifier (UUIDv4).
        patient_id: Foreign key to the assessed Patient.
        priority_level: Urgency classification (CRITICAL → LOW).
        clinical_summary: AI-generated summary of the patient's condition.
        ai_findings: Structured list of findings from vision + LLM analysis.
        recommended_actions: Suggested next steps for the treating physician.
        suggested_questions: Questions the AI recommends the physician ask.
        image_ids: References to the MedicalImage records used in analysis.
        note_ids: References to the ClinicalNote records used in analysis.
        model_metadata: Versions of the AI models used for traceability.
        generated_at: Timestamp of report creation.
    """

    id: UUID = field(default_factory=uuid4)
    patient_id: UUID | None = None
    priority_level: PriorityLevel = PriorityLevel.MEDIUM
    clinical_summary: str = ""
    ai_findings: list[AIFinding] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    suggested_questions: list[str] = field(default_factory=list)
    image_ids: list[UUID] = field(default_factory=list)
    note_ids: list[UUID] = field(default_factory=list)
    model_metadata: dict[str, str] = field(default_factory=dict)
    generated_at: datetime | None = None

    def to_document(self) -> dict:
        """Serialize to a MongoDB/Cosmos DB compatible document."""
        return {
            "_id": str(self.id),
            "patient_id": str(self.patient_id),
            "priority_level": self.priority_level.value,
            "clinical_summary": self.clinical_summary,
            "ai_findings": [
                {
                    "source": f.source,
                    "description": f.description,
                    "confidence": f.confidence,
                    "supporting_evidence": f.supporting_evidence,
                }
                for f in self.ai_findings
            ],
            "recommended_actions": self.recommended_actions,
            "suggested_questions": self.suggested_questions,
            "image_ids": [str(uid) for uid in self.image_ids],
            "note_ids": [str(uid) for uid in self.note_ids],
            "model_metadata": self.model_metadata,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
        }

    @classmethod
    def from_document(cls, doc: dict) -> TriageReport:
        """Deserialize from a MongoDB/Cosmos DB document."""
        return cls(
            id=UUID(doc["_id"]),
            patient_id=UUID(doc["patient_id"]),
            priority_level=PriorityLevel(doc["priority_level"]),
            clinical_summary=doc.get("clinical_summary", ""),
            ai_findings=[
                AIFinding(
                    source=f["source"],
                    description=f["description"],
                    confidence=f.get("confidence", 0.0),
                    supporting_evidence=f.get("supporting_evidence", ""),
                )
                for f in doc.get("ai_findings", [])
            ],
            recommended_actions=doc.get("recommended_actions", []),
            suggested_questions=doc.get("suggested_questions", []),
            image_ids=[UUID(uid) for uid in doc.get("image_ids", [])],
            note_ids=[UUID(uid) for uid in doc.get("note_ids", [])],
            model_metadata=doc.get("model_metadata", {}),
            generated_at=(
                datetime.fromisoformat(doc["generated_at"])
                if doc.get("generated_at")
                else None
            ),
        )
