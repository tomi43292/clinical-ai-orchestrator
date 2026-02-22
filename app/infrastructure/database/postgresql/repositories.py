"""
PostgreSQL Repository — Concrete implementation of the PatientRepository.

Handles the mapping between domain ``Patient`` entities and SQLAlchemy
``PatientModel`` ORM objects.  All queries are async and leverage
SQLAlchemy 2.0 select() style for explicit, optimisable SQL generation.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.domain.entities.clinical_note import ClinicalNote
from app.domain.entities.medical_image import (
    BodyRegion,
    ImageAnalysisResult,
    ImageModality,
    MedicalImage,
)
from app.domain.entities.patient import Gender, Patient
from app.domain.repositories.patient_repository import PatientRepository
from app.infrastructure.database.postgresql.models import (
    ClinicalNoteModel,
    MedicalImageModel,
    PatientModel,
)


class PostgresPatientRepository(PatientRepository):
    """
    SQLAlchemy-backed patient repository.

    Uses async sessions and explicit ``select()`` queries for full
    control over SQL generation and query optimisation.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ── Mapping helpers ─────────────────────────────────────────

    @staticmethod
    def _to_entity(model: PatientModel) -> Patient:
        """Map an ORM model to a domain entity."""
        return Patient(
            id=model.id,
            medical_record_number=model.medical_record_number,
            first_name=model.first_name,
            last_name=model.last_name,
            date_of_birth=model.date_of_birth,
            gender=Gender(model.gender),
            allergies=model.allergies or [],
            pre_existing_conditions=model.pre_existing_conditions or [],
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    @staticmethod
    def _to_model(entity: Patient) -> PatientModel:
        """Map a domain entity to an ORM model."""
        return PatientModel(
            id=entity.id,
            medical_record_number=entity.medical_record_number,
            first_name=entity.first_name,
            last_name=entity.last_name,
            date_of_birth=entity.date_of_birth,
            gender=entity.gender.value,
            allergies=entity.allergies,
            pre_existing_conditions=entity.pre_existing_conditions,
        )

    # ── CRUD operations ─────────────────────────────────────────

    async def create(self, patient: Patient) -> Patient:
        """Persist a new patient and return the entity with generated fields."""
        model = self._to_model(patient)
        self._session.add(model)
        await self._session.flush()
        await self._session.refresh(model)
        return self._to_entity(model)

    async def get_by_id(self, patient_id: UUID) -> Patient | None:
        """
        Fetch a patient by ID with eagerly loaded relationships.

        Uses ``selectinload`` to avoid N+1 queries when the caller
        accesses related images or notes.
        """
        stmt = (
            select(PatientModel)
            .options(
                selectinload(PatientModel.medical_images),
                selectinload(PatientModel.clinical_notes),
            )
            .where(PatientModel.id == patient_id)
        )
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def list_all(
        self,
        *,
        skip: int = 0,
        limit: int = 50,
        search: str | None = None,
    ) -> tuple[list[Patient], int]:
        """
        Paginated patient listing with optional name/MRN search.

        The search filter uses ``ILIKE`` for case-insensitive matching
        across first_name, last_name, and medical_record_number columns.
        """
        base_query = select(PatientModel)

        if search:
            search_filter = f"%{search}%"
            base_query = base_query.where(
                or_(
                    PatientModel.first_name.ilike(search_filter),
                    PatientModel.last_name.ilike(search_filter),
                    PatientModel.medical_record_number.ilike(search_filter),
                )
            )

        # Count total matching records
        count_stmt = select(func.count()).select_from(base_query.subquery())
        total = (await self._session.execute(count_stmt)).scalar() or 0

        # Fetch paginated results
        stmt = (
            base_query
            .order_by(PatientModel.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        models = result.scalars().all()

        return [self._to_entity(m) for m in models], total

    async def update(self, patient: Patient) -> Patient:
        """Merge updated fields into the existing patient record."""
        stmt = select(PatientModel).where(PatientModel.id == patient.id)
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()

        if model is None:
            raise ValueError(f"Patient {patient.id} not found for update.")

        model.first_name = patient.first_name
        model.last_name = patient.last_name
        model.date_of_birth = patient.date_of_birth
        model.gender = patient.gender.value
        model.allergies = patient.allergies
        model.pre_existing_conditions = patient.pre_existing_conditions

        await self._session.flush()
        await self._session.refresh(model)
        return self._to_entity(model)

    async def delete(self, patient_id: UUID) -> bool:
        """Delete a patient by ID. Returns True if the row existed."""
        stmt = select(PatientModel).where(PatientModel.id == patient_id)
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()

        if model is None:
            return False

        await self._session.delete(model)
        await self._session.flush()
        return True


class PostgresMedicalImageRepository:
    """
    Repository for ``MedicalImage`` entities in PostgreSQL.

    Handles image metadata persistence and analysis result storage.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, image: MedicalImage) -> MedicalImage:
        """Persist a new medical image record."""
        model = MedicalImageModel(
            id=image.id,
            patient_id=image.patient_id,
            filename=image.filename,
            content_type=image.content_type,
            storage_path=image.storage_path,
            modality=image.modality.value,
            body_region=image.body_region.value,
            analysis_result=None,
        )
        self._session.add(model)
        await self._session.flush()
        await self._session.refresh(model)
        return self._model_to_entity(model)

    async def update_analysis_result(
        self, image_id: UUID, analysis: dict
    ) -> None:
        """Store the Computer Vision analysis result for an image."""
        stmt = select(MedicalImageModel).where(MedicalImageModel.id == image_id)
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()
        if model:
            model.analysis_result = analysis
            await self._session.flush()

    async def get_by_patient_id(self, patient_id: UUID) -> list[MedicalImage]:
        """Retrieve all images for a patient, ordered by creation date."""
        stmt = (
            select(MedicalImageModel)
            .where(MedicalImageModel.patient_id == patient_id)
            .order_by(MedicalImageModel.created_at.desc())
        )
        result = await self._session.execute(stmt)
        return [self._model_to_entity(m) for m in result.scalars().all()]

    @staticmethod
    def _model_to_entity(model: MedicalImageModel) -> MedicalImage:
        analysis = None
        if model.analysis_result:
            from app.domain.entities.medical_image import DetectedAnomaly

            anomalies = [
                DetectedAnomaly(
                    label=a["label"],
                    confidence=a["confidence"],
                    region=a["region"],
                    bounding_box=a.get("bounding_box"),
                )
                for a in model.analysis_result.get("anomalies", [])
            ]
            analysis = ImageAnalysisResult(
                anomalies=anomalies,
                description=model.analysis_result.get("description", ""),
                raw_tags=model.analysis_result.get("raw_tags", []),
                model_version=model.analysis_result.get("model_version", ""),
            )
        return MedicalImage(
            id=model.id,
            patient_id=model.patient_id,
            filename=model.filename,
            content_type=model.content_type,
            storage_path=model.storage_path,
            modality=ImageModality(model.modality),
            body_region=BodyRegion(model.body_region),
            analysis_result=analysis,
            created_at=model.created_at,
        )


class PostgresClinicalNoteRepository:
    """Repository for ``ClinicalNote`` entities in PostgreSQL."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, note: ClinicalNote) -> ClinicalNote:
        """Persist a new clinical note."""
        model = ClinicalNoteModel(
            id=note.id,
            patient_id=note.patient_id,
            author=note.author,
            content=note.content,
            note_type=note.note_type,
        )
        self._session.add(model)
        await self._session.flush()
        await self._session.refresh(model)
        return self._model_to_entity(model)

    async def get_by_patient_id(self, patient_id: UUID) -> list[ClinicalNote]:
        """Retrieve all clinical notes for a patient."""
        stmt = (
            select(ClinicalNoteModel)
            .where(ClinicalNoteModel.patient_id == patient_id)
            .order_by(ClinicalNoteModel.created_at.desc())
        )
        result = await self._session.execute(stmt)
        return [self._model_to_entity(m) for m in result.scalars().all()]

    @staticmethod
    def _model_to_entity(model: ClinicalNoteModel) -> ClinicalNote:
        return ClinicalNote(
            id=model.id,
            patient_id=model.patient_id,
            author=model.author,
            content=model.content,
            note_type=model.note_type,
            created_at=model.created_at,
        )
