"""
API v1 Router — Aggregates all versioned endpoint modules.
"""

from fastapi import APIRouter

from app.api.v1.clinical_notes import router as clinical_notes_router
from app.api.v1.medical_images import router as medical_images_router
from app.api.v1.patients import router as patients_router
from app.api.v1.search import router as search_router
from app.api.v1.triage import router as triage_router

api_v1_router = APIRouter()

api_v1_router.include_router(patients_router)
api_v1_router.include_router(medical_images_router)
api_v1_router.include_router(clinical_notes_router)
api_v1_router.include_router(triage_router)
api_v1_router.include_router(search_router)
