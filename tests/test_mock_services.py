"""
Unit Tests for Mock Services.

Verifies that mock implementations produce valid outputs matching
the expected domain contracts.
"""

from uuid import uuid4

import pytest
import pytest_asyncio

from app.domain.entities.medical_image import BodyRegion, ImageModality
from app.domain.entities.patient import Gender, Patient
from app.domain.entities.clinical_note import ClinicalNote
from app.infrastructure.azure.mock_services import (
    MockLLMService,
    MockSearchService,
    MockVisionService,
)


class TestMockVisionService:
    """Tests for the MockVisionService implementation."""

    @pytest.mark.asyncio
    async def test_analyze_image_returns_valid_result(self):
        service = MockVisionService()
        result = await service.analyze_image(
            image_bytes=b"fake_image_data",
            modality=ImageModality.XRAY,
            body_region="chest",
        )
        assert result.description
        assert len(result.anomalies) > 0
        assert result.model_version == "mock-vision-1.0"
        assert all(0.0 <= a.confidence <= 1.0 for a in result.anomalies)

    @pytest.mark.asyncio
    async def test_health_check(self):
        service = MockVisionService()
        assert await service.health_check() is True


class TestMockLLMService:
    """Tests for the MockLLMService implementation."""

    @pytest.mark.asyncio
    async def test_generate_triage_returns_report(self):
        service = MockLLMService()
        patient = Patient(
            id=uuid4(),
            medical_record_number="MRN-TEST",
            first_name="Test",
            last_name="Patient",
            gender=Gender.MALE,
        )
        report = await service.generate_triage(
            patient=patient,
            image_analyses=[],
            clinical_notes=[],
        )
        assert report.patient_id == patient.id
        assert report.priority_level is not None
        assert report.clinical_summary
        assert len(report.ai_findings) > 0

    @pytest.mark.asyncio
    async def test_health_check(self):
        service = MockLLMService()
        assert await service.health_check() is True


class TestMockSearchService:
    """Tests for the MockSearchService implementation."""

    @pytest.mark.asyncio
    async def test_index_and_search(self):
        service = MockSearchService()
        await service.index_document(
            document_id="doc-1",
            content="Patient with acute respiratory distress",
            metadata={"type": "note"},
        )
        results = await service.search(query="respiratory")
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_search_empty_query(self):
        service = MockSearchService()
        results = await service.search(query="nonexistent_term_xyz")
        # Mock may still return results; just verify no crash
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_health_check(self):
        service = MockSearchService()
        assert await service.health_check() is True
