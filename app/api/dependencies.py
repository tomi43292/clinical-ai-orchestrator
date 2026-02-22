"""
API Dependencies — Dependency injection container for FastAPI.

Provides factory functions that wire together domain interfaces with
their concrete infrastructure implementations.  The ``SERVICE_MODE``
setting controls whether real Azure services or local mocks are used.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import ServiceMode, get_settings
from app.domain.repositories.patient_repository import PatientRepository
from app.domain.repositories.report_repository import TriageReportRepository
from app.domain.services.llm_service import ClinicalLLMService
from app.domain.services.search_service import ClinicalSearchService
from app.domain.services.vision_service import VisionAnalysisService
from app.infrastructure.database.cosmosdb.repositories import CosmosTriageReportRepository
from app.infrastructure.database.postgresql.repositories import (
    PostgresClinicalNoteRepository,
    PostgresMedicalImageRepository,
    PostgresPatientRepository,
)
from app.infrastructure.database.postgresql.session import get_db_session

# ── Type alias for injected session ─────────────────────────────
DBSession = Annotated[AsyncSession, Depends(get_db_session)]


# ── Repository factories ────────────────────────────────────────

def get_patient_repo(session: DBSession) -> PatientRepository:
    """Provide a patient repository bound to the current session."""
    return PostgresPatientRepository(session)


def get_image_repo(session: DBSession) -> PostgresMedicalImageRepository:
    """Provide a medical image repository bound to the current session."""
    return PostgresMedicalImageRepository(session)


def get_note_repo(session: DBSession) -> PostgresClinicalNoteRepository:
    """Provide a clinical note repository bound to the current session."""
    return PostgresClinicalNoteRepository(session)


def get_report_repo() -> TriageReportRepository:
    """Provide a triage report repository (Cosmos DB)."""
    return CosmosTriageReportRepository()


# ── Service factories (mode-aware) ──────────────────────────────

_vision_service: VisionAnalysisService | None = None
_llm_service: ClinicalLLMService | None = None
_search_service: ClinicalSearchService | None = None


def get_vision_service() -> VisionAnalysisService:
    """
    Provide the vision analysis service.

    Returns the Azure implementation or the mock depending on
    ``SERVICE_MODE``.
    """
    global _vision_service
    if _vision_service is None:
        settings = get_settings()
        if settings.service_mode == ServiceMode.AZURE:
            from app.infrastructure.azure.vision_client import AzureVisionService
            _vision_service = AzureVisionService()
        else:
            from app.infrastructure.azure.mock_services import MockVisionService
            _vision_service = MockVisionService()
    return _vision_service


def get_llm_service() -> ClinicalLLMService:
    """
    Provide the clinical LLM service.

    Returns the LangChain agent or the mock depending on ``SERVICE_MODE``.
    """
    global _llm_service
    if _llm_service is None:
        settings = get_settings()
        if settings.service_mode == ServiceMode.AZURE:
            from app.infrastructure.langchain.triage_agent import LangChainTriageAgent
            _llm_service = LangChainTriageAgent()
        else:
            from app.infrastructure.azure.mock_services import MockLLMService
            _llm_service = MockLLMService()
    return _llm_service


def get_search_service() -> ClinicalSearchService:
    """
    Provide the clinical search service.

    Returns the Azure Cognitive Search client or an in-memory mock.
    """
    global _search_service
    if _search_service is None:
        settings = get_settings()
        if settings.service_mode == ServiceMode.AZURE:
            from app.infrastructure.azure.search_client import AzureCognitiveSearchService
            _search_service = AzureCognitiveSearchService()
        else:
            from app.infrastructure.azure.mock_services import MockSearchService
            _search_service = MockSearchService()
    return _search_service
