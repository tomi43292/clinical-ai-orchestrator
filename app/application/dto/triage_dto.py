"""
Triage DTOs — Pydantic schemas for triage request/response operations.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class TriageRequest(BaseModel):
    """
    Request body for generating a triage assessment.

    If ``image_ids`` or ``note_ids`` are omitted, the system will use
    all available images and notes for the patient.
    """

    image_ids: list[UUID] | None = Field(
        None, description="Specific image IDs to include (all if omitted)"
    )
    note_ids: list[UUID] | None = Field(
        None, description="Specific note IDs to include (all if omitted)"
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "image_ids": None,
            "note_ids": None,
        }
    })


class AIFindingResponse(BaseModel):
    """A single AI finding in the triage report response."""

    source: str
    description: str
    confidence: float
    supporting_evidence: str


class TriageReportResponse(BaseModel):
    """Full triage report response."""

    id: UUID
    patient_id: UUID
    priority_level: str
    clinical_summary: str
    ai_findings: list[AIFindingResponse]
    recommended_actions: list[str]
    suggested_questions: list[str]
    model_metadata: dict[str, str]
    generated_at: datetime | None

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "id": "c3d4e5f6-a7b8-9012-cdef-123456789012",
            "patient_id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
            "priority_level": "high",
            "clinical_summary": "High-priority findings identified...",
            "ai_findings": [
                {
                    "source": "vision",
                    "description": "Detected 'pulmonary_opacity' in region 'right_lower_lobe'",
                    "confidence": 0.87,
                    "supporting_evidence": "Possible opacification in the right lower lobe.",
                }
            ],
            "recommended_actions": ["Review with radiologist", "Further diagnostic workup"],
            "suggested_questions": [
                "Has the patient experienced similar symptoms before?",
            ],
            "model_metadata": {"vision_model": "azure-cv-4.0", "llm_model": "gpt-4"},
            "generated_at": "2024-02-22T15:30:00Z",
        }
    })


class ClinicalNoteCreate(BaseModel):
    """Schema for adding a clinical note to a patient."""

    author: str = Field(..., min_length=1, max_length=150)
    content: str = Field(..., min_length=1, description="Clinical observation text")
    note_type: str = Field("progress", pattern="^(admission|progress|discharge|emergency)$")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "author": "Dr. Martínez",
            "content": "Patient presents with persistent dry cough and fever (38.5°C) for 3 days. Bilateral crackles on auscultation.",
            "note_type": "admission",
        }
    })


class ClinicalNoteResponse(BaseModel):
    """Clinical note in API responses."""

    id: UUID
    patient_id: UUID
    author: str
    content: str
    note_type: str
    created_at: datetime | None

    model_config = ConfigDict(from_attributes=True)
