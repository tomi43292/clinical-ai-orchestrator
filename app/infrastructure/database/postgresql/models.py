"""
SQLAlchemy ORM Models — Relational schema for clinical data in PostgreSQL.

These models implement the physical database schema and provide bidirectional
mapping to domain entities.  Relationships use lazy loading by default but
expose ``selectinload`` options for optimized query patterns.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime

from sqlalchemy import (
    ARRAY,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Index,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

    pass


class PatientModel(Base):
    """
    ORM model for the ``patients`` table.

    Stores demographic and medical history data in a relational format
    optimized for search and join operations.
    """

    __tablename__ = "patients"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    medical_record_number: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False, index=True
    )
    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    date_of_birth: Mapped[date | None] = mapped_column(nullable=True)
    gender: Mapped[str] = mapped_column(
        SAEnum("male", "female", "other", "unknown", name="gender_enum"),
        default="unknown",
    )
    allergies: Mapped[list[str]] = mapped_column(
        ARRAY(String), default=list, server_default="{}"
    )
    pre_existing_conditions: Mapped[list[str]] = mapped_column(
        ARRAY(String), default=list, server_default="{}"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # ── Relationships ───────────────────────────────────────────
    medical_images: Mapped[list[MedicalImageModel]] = relationship(
        back_populates="patient", cascade="all, delete-orphan", lazy="selectin"
    )
    clinical_notes: Mapped[list[ClinicalNoteModel]] = relationship(
        back_populates="patient", cascade="all, delete-orphan", lazy="selectin"
    )

    # ── Indexes for optimized queries ───────────────────────────
    __table_args__ = (
        Index("ix_patients_name", "last_name", "first_name"),
        Index("ix_patients_created", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Patient(id={self.id}, mrn={self.medical_record_number})>"


class MedicalImageModel(Base):
    """
    ORM model for the ``medical_images`` table.

    Stores image metadata and the structured JSON output of the Computer
    Vision analysis.  The actual image bytes are stored externally (blob
    storage); only the ``storage_path`` is persisted here.
    """

    __tablename__ = "medical_images"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(50), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(500), nullable=False)
    modality: Mapped[str] = mapped_column(
        SAEnum("xray", "ct", "mri", "ultrasound", "other", name="modality_enum"),
        default="other",
    )
    body_region: Mapped[str] = mapped_column(
        SAEnum(
            "chest", "abdomen", "head", "spine", "extremities", "pelvis", "other",
            name="body_region_enum",
        ),
        default="other",
    )
    analysis_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # ── Relationships ───────────────────────────────────────────
    patient: Mapped[PatientModel] = relationship(back_populates="medical_images")

    # ── Indexes ─────────────────────────────────────────────────
    __table_args__ = (
        Index("ix_medical_images_patient", "patient_id"),
        Index("ix_medical_images_modality", "modality"),
    )

    def __repr__(self) -> str:
        return f"<MedicalImage(id={self.id}, modality={self.modality})>"


class ClinicalNoteModel(Base):
    """
    ORM model for the ``clinical_notes`` table.

    Stores free-text clinical observations linked to a patient.
    Supports full-text search via PostgreSQL's ``tsvector`` if needed.
    """

    __tablename__ = "clinical_notes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False
    )
    author: Mapped[str] = mapped_column(String(150), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    note_type: Mapped[str] = mapped_column(String(50), default="progress")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # ── Relationships ───────────────────────────────────────────
    patient: Mapped[PatientModel] = relationship(back_populates="clinical_notes")

    # ── Indexes ─────────────────────────────────────────────────
    __table_args__ = (
        Index("ix_clinical_notes_patient", "patient_id"),
        Index("ix_clinical_notes_created", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<ClinicalNote(id={self.id}, author={self.author})>"
