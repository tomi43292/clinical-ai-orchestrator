"""
Clinical Note Entity — Free-text clinical observations recorded by clinicians.

These notes feed into the LLM reasoning pipeline alongside image analysis
results to produce the final triage assessment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class ClinicalNote:
    """
    Domain entity for a clinical note attached to a patient.

    Attributes:
        id: Unique identifier (UUIDv4).
        patient_id: Foreign key to the owning Patient.
        author: Name or identifier of the clinician who authored the note.
        content: Free-text clinical observations.
        note_type: Category (e.g., "admission", "progress", "discharge").
        created_at: Timestamp when the note was written.
    """

    id: UUID = field(default_factory=uuid4)
    patient_id: UUID | None = None
    author: str = ""
    content: str = ""
    note_type: str = "progress"
    created_at: datetime | None = None
