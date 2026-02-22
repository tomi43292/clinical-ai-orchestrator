"""
Analyze Medical Image Use Case — Processes uploaded images through CV pipeline.

Receives image bytes, persists metadata, invokes the Computer Vision
service, and stores the structured analysis output.
"""

from __future__ import annotations

import logging
import os
from uuid import UUID, uuid4

from app.domain.entities.medical_image import BodyRegion, ImageModality, MedicalImage
from app.domain.services.vision_service import VisionAnalysisService
from app.infrastructure.database.postgresql.repositories import PostgresMedicalImageRepository

logger = logging.getLogger(__name__)

# Local upload directory for development
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "uploads")


class AnalyzeMedicalImageUseCase:
    """
    Orchestrates image upload, CV analysis, and result persistence.

    Steps:
    1. Save the image bytes to local storage (or blob storage in production).
    2. Persist image metadata in PostgreSQL.
    3. Send the image to the Vision Analysis Service.
    4. Update the database record with the analysis results.
    """

    def __init__(
        self,
        image_repo: PostgresMedicalImageRepository,
        vision_service: VisionAnalysisService,
    ) -> None:
        self._image_repo = image_repo
        self._vision_service = vision_service

    async def execute(
        self,
        patient_id: UUID,
        filename: str,
        content_type: str,
        image_bytes: bytes,
        modality: str = "other",
        body_region: str = "other",
    ) -> MedicalImage:
        """
        Process and analyze a medical image.

        Args:
            patient_id: UUID of the patient this image belongs to.
            filename: Original filename.
            content_type: MIME type (e.g., ``image/png``).
            image_bytes: Raw image data.
            modality: Imaging modality string.
            body_region: Anatomical region string.

        Returns:
            ``MedicalImage`` entity with populated analysis results.
        """
        image_id = uuid4()

        # 1. Save image to storage
        storage_path = await self._save_image(image_id, filename, image_bytes)

        # 2. Create image record
        image_modality = ImageModality(modality)
        image_body_region = BodyRegion(body_region)

        image = MedicalImage(
            id=image_id,
            patient_id=patient_id,
            filename=filename,
            content_type=content_type,
            storage_path=storage_path,
            modality=image_modality,
            body_region=image_body_region,
        )

        persisted = await self._image_repo.create(image)

        # 3. Analyze with Computer Vision
        logger.info("Analyzing image '%s' (modality=%s)...", filename, modality)
        analysis = await self._vision_service.analyze_image(
            image_bytes=image_bytes,
            modality=image_modality,
            body_region=body_region,
        )

        # 4. Store analysis results
        analysis_dict = {
            "anomalies": [
                {
                    "label": a.label,
                    "confidence": a.confidence,
                    "region": a.region,
                    "bounding_box": a.bounding_box,
                }
                for a in analysis.anomalies
            ],
            "description": analysis.description,
            "raw_tags": analysis.raw_tags,
            "model_version": analysis.model_version,
        }
        await self._image_repo.update_analysis_result(image_id, analysis_dict)

        logger.info(
            "Image analysis complete: %d anomalies detected in '%s'.",
            len(analysis.anomalies),
            filename,
        )

        # Return the image with analysis populated
        return MedicalImage(
            id=persisted.id,
            patient_id=persisted.patient_id,
            filename=persisted.filename,
            content_type=persisted.content_type,
            storage_path=persisted.storage_path,
            modality=persisted.modality,
            body_region=persisted.body_region,
            analysis_result=analysis,
            created_at=persisted.created_at,
        )

    @staticmethod
    async def _save_image(image_id: UUID, filename: str, image_bytes: bytes) -> str:
        """Save image bytes to local storage. Returns the storage path."""
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        ext = os.path.splitext(filename)[1] or ".bin"
        path = os.path.join(UPLOAD_DIR, f"{image_id}{ext}")
        with open(path, "wb") as f:
            f.write(image_bytes)
        return path
