"""
Patient DTOs — Pydantic schemas for patient API operations.

Separates input validation (``Create``, ``Update``) from output
serialization (``Response``, ``ListResponse``) to enforce the single
responsibility principle at the schema level.
"""

from __future__ import annotations

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class PatientCreate(BaseModel):
    """Schema for creating a new patient record."""

    medical_record_number: str = Field(
        ..., min_length=1, max_length=50, description="External medical record number"
    )
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    date_of_birth: date | None = Field(None, description="Patient date of birth")
    gender: str = Field("unknown", pattern="^(male|female|other|unknown)$")
    allergies: list[str] = Field(default_factory=list)
    pre_existing_conditions: list[str] = Field(default_factory=list)

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "medical_record_number": "MRN-2024-001",
            "first_name": "María",
            "last_name": "García López",
            "date_of_birth": "1985-03-15",
            "gender": "female",
            "allergies": ["Penicillin"],
            "pre_existing_conditions": ["Type 2 Diabetes", "Hypertension"],
        }
    })


class PatientUpdate(BaseModel):
    """Schema for updating an existing patient (partial updates)."""

    first_name: str | None = None
    last_name: str | None = None
    date_of_birth: date | None = None
    gender: str | None = Field(None, pattern="^(male|female|other|unknown)$")
    allergies: list[str] | None = None
    pre_existing_conditions: list[str] | None = None


class PatientResponse(BaseModel):
    """Schema for a single patient in API responses."""

    id: UUID
    medical_record_number: str
    first_name: str
    last_name: str
    full_name: str
    date_of_birth: date | None
    age: int | None
    gender: str
    allergies: list[str]
    pre_existing_conditions: list[str]
    created_at: datetime | None
    updated_at: datetime | None

    model_config = ConfigDict(from_attributes=True)


class PatientListResponse(BaseModel):
    """Paginated list of patients."""

    items: list[PatientResponse]
    total: int
    skip: int
    limit: int
