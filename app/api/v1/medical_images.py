"""
Medical Image Endpoints — Upload and analyze diagnostic images.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from app.api.dependencies import (
    DBSession,
    get_image_repo,
    get_patient_repo,
    get_vision_service,
)
from app.application.dto.image_dto import (
    DetectedAnomalyResponse,
    ImageAnalysisResponse,
    ImageUploadResponse,
)
from app.application.use_cases.analyze_image import AnalyzeMedicalImageUseCase
from app.domain.exceptions import PatientNotFoundError
from app.domain.repositories.patient_repository import PatientRepository
from app.domain.services.vision_service import VisionAnalysisService
from app.infrastructure.database.postgresql.repositories import PostgresMedicalImageRepository

router = APIRouter(prefix="/patients/{patient_id}/medical-images", tags=["Medical Images"])


@router.post(
    "",
    response_model=ImageAnalysisResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and analyze a medical image",
    description=(
        "Uploads a diagnostic image (X-Ray, CT, MRI, etc.), sends it through "
        "the Computer Vision pipeline for anomaly detection, and returns "
        "structured analysis results."
    ),
)
async def upload_and_analyze_image(
    patient_id: UUID,
    patient_repo: Annotated[PatientRepository, Depends(get_patient_repo)],
    image_repo: Annotated[PostgresMedicalImageRepository, Depends(get_image_repo)],
    vision_service: Annotated[VisionAnalysisService, Depends(get_vision_service)],
    file: UploadFile = File(..., description="Medical image file"),
    modality: str = Form("other", description="Imaging modality: xray, ct, mri, ultrasound, other"),
    body_region: str = Form("other", description="Body region: chest, abdomen, head, spine, extremities, pelvis, other"),
) -> ImageAnalysisResponse:
    """Upload a medical image and run Computer Vision analysis."""

    # Verify patient exists
    patient = await patient_repo.get_by_id(patient_id)
    if patient is None:
        raise PatientNotFoundError(str(patient_id))

    # Read image bytes
    image_bytes = await file.read()

    # Execute use case
    use_case = AnalyzeMedicalImageUseCase(image_repo, vision_service)
    result = await use_case.execute(
        patient_id=patient_id,
        filename=file.filename or "unnamed.bin",
        content_type=file.content_type or "application/octet-stream",
        image_bytes=image_bytes,
        modality=modality,
        body_region=body_region,
    )

    # Map to response
    anomalies = []
    if result.analysis_result:
        anomalies = [
            DetectedAnomalyResponse(
                label=a.label,
                confidence=a.confidence,
                region=a.region,
                bounding_box=a.bounding_box,
            )
            for a in result.analysis_result.anomalies
        ]

    return ImageAnalysisResponse(
        image_id=result.id,
        patient_id=result.patient_id,
        filename=result.filename,
        modality=result.modality.value,
        description=result.analysis_result.description if result.analysis_result else "",
        anomalies=anomalies,
        raw_tags=result.analysis_result.raw_tags if result.analysis_result else [],
        model_version=result.analysis_result.model_version if result.analysis_result else "",
    )


@router.get(
    "",
    response_model=list[ImageUploadResponse],
    summary="List images for a patient",
    description="Retrieve all medical images associated with a patient.",
)
async def list_patient_images(
    patient_id: UUID,
    image_repo: Annotated[PostgresMedicalImageRepository, Depends(get_image_repo)],
) -> list[ImageUploadResponse]:
    images = await image_repo.get_by_patient_id(patient_id)
    return [
        ImageUploadResponse(
            id=img.id,
            patient_id=img.patient_id,
            filename=img.filename,
            modality=img.modality.value,
            body_region=img.body_region.value,
            created_at=img.created_at,
        )
        for img in images
    ]
