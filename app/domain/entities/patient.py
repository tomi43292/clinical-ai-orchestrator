"""
Patient Entity — Core domain representation of a patient record.

Encapsulates demographic and medical data that lives in the relational
store (PostgreSQL).  This entity is persistence-agnostic: it carries no
ORM dependency and can be serialised to any transport format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from uuid import UUID, uuid4


class Gender(str, Enum):
    """Biological sex classification for clinical context."""

    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Patient:
    """
    Immutable domain entity representing a patient.

    Attributes:
        id: Unique identifier (UUIDv4).
        medical_record_number: External medical-record number (MRN).
        first_name: Patient's first name.
        last_name: Patient's last name.
        date_of_birth: Date of birth for age-based triage adjustments.
        gender: Biological sex relevant to diagnostic reasoning.
        allergies: Known allergies (free-text list).
        pre_existing_conditions: Pre-existing medical conditions.
        created_at: Timestamp of record creation.
        updated_at: Timestamp of last modification.
    """

    id: UUID = field(default_factory=uuid4)
    medical_record_number: str = ""
    first_name: str = ""
    last_name: str = ""
    date_of_birth: date | None = None
    gender: Gender = Gender.UNKNOWN
    allergies: list[str] = field(default_factory=list)
    pre_existing_conditions: list[str] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def full_name(self) -> str:
        """Return the patient's full name."""
        return f"{self.first_name} {self.last_name}".strip()

    @property
    def age(self) -> int | None:
        """Calculate age in years from date_of_birth, or None if unknown."""
        if self.date_of_birth is None:
            return None
        today = date.today()
        delta = today.year - self.date_of_birth.year
        if (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day):
            delta -= 1
        return delta
