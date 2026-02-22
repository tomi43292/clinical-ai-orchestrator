"""
Azure Cognitive Services — Computer Vision Client.

Wraps Azure's Computer Vision API (Image Analysis 4.0) to detect
medical anomalies in diagnostic images.  Implements the domain
``VisionAnalysisService`` interface.
"""

from __future__ import annotations

import logging
from typing import Any

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

from app.core.config import get_settings
from app.domain.entities.medical_image import (
    DetectedAnomaly,
    ImageAnalysisResult,
    ImageModality,
)
from app.domain.exceptions import ImageAnalysisError
from app.domain.services.vision_service import VisionAnalysisService

logger = logging.getLogger(__name__)


class AzureVisionService(VisionAnalysisService):
    """
    Azure Cognitive Services implementation of ``VisionAnalysisService``.

    Connects to the Azure Computer Vision Image Analysis endpoint and
    extracts tags, captions, and object detections from medical images.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._client = ImageAnalysisClient(
            endpoint=settings.azure_vision_endpoint,
            credential=AzureKeyCredential(settings.azure_vision_api_key),
        )

    async def analyze_image(
        self,
        image_bytes: bytes,
        modality: ImageModality,
        body_region: str = "unknown",
    ) -> ImageAnalysisResult:
        """
        Send image bytes to Azure Computer Vision and parse the response.

        Requests caption, tags, and object detection features.  Each
        detected object is mapped to a ``DetectedAnomaly`` for downstream
        clinical reasoning.
        """
        try:
            result = self._client.analyze(
                image_data=image_bytes,
                visual_features=[
                    VisualFeatures.CAPTION,
                    VisualFeatures.TAGS,
                    VisualFeatures.OBJECTS,
                    VisualFeatures.DENSE_CAPTIONS,
                ],
            )

            anomalies = self._extract_anomalies(result)
            description = self._build_description(result)
            raw_tags = self._extract_tags(result)

            logger.info(
                "Azure Vision analysis complete — %d anomalies detected.",
                len(anomalies),
            )

            return ImageAnalysisResult(
                anomalies=anomalies,
                description=description,
                raw_tags=raw_tags,
                model_version=getattr(result, "model_version", "azure-cv-4.0"),
            )

        except Exception as exc:
            logger.error("Azure Vision analysis failed: %s", str(exc))
            raise ImageAnalysisError(reason=str(exc)) from exc

    async def health_check(self) -> bool:
        """Verify the Azure Vision endpoint is reachable."""
        try:
            # A lightweight operation to test connectivity
            return self._client is not None
        except Exception:
            return False

    # ── Private helpers ─────────────────────────────────────────

    @staticmethod
    def _extract_anomalies(result: Any) -> list[DetectedAnomaly]:
        """Map detected objects to domain ``DetectedAnomaly`` instances."""
        anomalies: list[DetectedAnomaly] = []

        if hasattr(result, "objects") and result.objects:
            for obj in result.objects.list:
                bbox = obj.bounding_box
                anomalies.append(
                    DetectedAnomaly(
                        label=obj.tags[0].name if obj.tags else "unknown",
                        confidence=obj.tags[0].confidence if obj.tags else 0.0,
                        region=f"x:{bbox.x}, y:{bbox.y}, w:{bbox.width}, h:{bbox.height}",
                        bounding_box=[bbox.x, bbox.y, bbox.width, bbox.height],
                    )
                )

        return anomalies

    @staticmethod
    def _build_description(result: Any) -> str:
        """Extract the primary caption as the image description."""
        if hasattr(result, "caption") and result.caption:
            return result.caption.text
        return "No description available."

    @staticmethod
    def _extract_tags(result: Any) -> list[str]:
        """Collect all tags as a flat list of strings."""
        if hasattr(result, "tags") and result.tags:
            return [tag.name for tag in result.tags.list]
        return []
