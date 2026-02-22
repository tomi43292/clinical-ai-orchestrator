"""
Unit Tests for Domain Entities.

Validates entity construction, computed properties, serialization,
and edge-case handling.
"""

from datetime import date, datetime, timezone
from uuid import uuid4

import pytest

from app.domain.entities.patient import Gender, Patient
from app.domain.entities.medical_image import (
    BodyRegion,
    DetectedAnomaly,
    ImageAnalysisResult,
    ImageModality,
    MedicalImage,
)
from app.domain.entities.clinical_note import ClinicalNote
from app.domain.entities.triage_report import (
    AIFinding,
    PriorityLevel,
    TriageReport,
)


class TestPatientEntity:
    """Tests for the Patient domain entity."""

    def test_full_name(self):
        patient = Patient(
            id=uuid4(),
            medical_record_number="MRN-001",
            first_name="María",
            last_name="García",
            gender=Gender.FEMALE,
        )
        assert patient.full_name == "María García"

    def test_age_calculation(self):
        patient = Patient(
            id=uuid4(),
            medical_record_number="MRN-002",
            first_name="Juan",
            last_name="López",
            date_of_birth=date(1990, 1, 1),
            gender=Gender.MALE,
        )
        assert patient.age is not None
        assert patient.age >= 34  # Will be at least 34 as of 2024

    def test_age_none_without_dob(self):
        patient = Patient(
            id=uuid4(),
            medical_record_number="MRN-003",
            first_name="Test",
            last_name="User",
            gender=Gender.UNKNOWN,
        )
        assert patient.age is None

    def test_default_values(self):
        patient = Patient(
            id=uuid4(),
            medical_record_number="MRN-004",
            first_name="Ana",
            last_name="Martínez",
            gender=Gender.FEMALE,
        )
        assert patient.allergies == []
        assert patient.pre_existing_conditions == []


class TestMedicalImageEntity:
    """Tests for the MedicalImage domain entity."""

    def test_image_creation(self):
        image = MedicalImage(
            id=uuid4(),
            patient_id=uuid4(),
            filename="chest_xray.png",
            content_type="image/png",
            storage_path="/uploads/test.png",
            modality=ImageModality.XRAY,
            body_region=BodyRegion.CHEST,
        )
        assert image.filename == "chest_xray.png"
        assert image.modality == ImageModality.XRAY
        assert image.analysis_result is None

    def test_image_with_analysis(self):
        analysis = ImageAnalysisResult(
            anomalies=[
                DetectedAnomaly(
                    label="opacity",
                    confidence=0.85,
                    region="right_lower_lobe",
                )
            ],
            description="Opacity detected",
            raw_tags=["opacity", "effusion"],
            model_version="test-1.0",
        )
        image = MedicalImage(
            id=uuid4(),
            patient_id=uuid4(),
            filename="test.png",
            content_type="image/png",
            storage_path="/test",
            modality=ImageModality.CT,
            body_region=BodyRegion.CHEST,
            analysis_result=analysis,
        )
        assert len(image.analysis_result.anomalies) == 1
        assert image.analysis_result.anomalies[0].confidence == 0.85


class TestTriageReportEntity:
    """Tests for the TriageReport domain entity."""

    def test_report_creation(self):
        report = TriageReport(
            id=uuid4(),
            patient_id=uuid4(),
            priority_level=PriorityLevel.HIGH,
            clinical_summary="Critical findings detected.",
            ai_findings=[
                AIFinding(
                    source="vision",
                    description="Opacity in RLL",
                    confidence=0.87,
                    supporting_evidence="CV analysis",
                )
            ],
            recommended_actions=["Consult radiologist"],
            generated_at=datetime.now(tz=timezone.utc),
        )
        assert report.priority_level == PriorityLevel.HIGH
        assert len(report.ai_findings) == 1
        assert "Consult radiologist" in report.recommended_actions

    def test_report_to_mongo_dict(self):
        report_id = uuid4()
        patient_id = uuid4()
        report = TriageReport(
            id=report_id,
            patient_id=patient_id,
            priority_level=PriorityLevel.CRITICAL,
            clinical_summary="Urgent assessment required.",
            ai_findings=[],
            generated_at=datetime.now(tz=timezone.utc),
        )
        mongo_dict = report.to_mongo_dict()
        assert mongo_dict["_id"] == str(report_id)
        assert mongo_dict["patient_id"] == str(patient_id)
        assert mongo_dict["priority_level"] == "critical"

    def test_priority_levels(self):
        assert PriorityLevel.CRITICAL.value == "critical"
        assert PriorityLevel.HIGH.value == "high"
        assert PriorityLevel.MEDIUM.value == "medium"
        assert PriorityLevel.LOW.value == "low"


class TestClinicalNoteEntity:
    """Tests for the ClinicalNote domain entity."""

    def test_note_creation(self):
        note = ClinicalNote(
            id=uuid4(),
            patient_id=uuid4(),
            author="Dr. Rodríguez",
            content="Patient presents with fever and dry cough.",
            note_type="admission",
        )
        assert note.author == "Dr. Rodríguez"
        assert note.note_type == "admission"
