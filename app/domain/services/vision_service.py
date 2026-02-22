"""
Vision Analysis Service — Abstract interface for Computer Vision operations.

Defines the contract for any image analysis provider, whether it is
Azure Cognitive Services, a local mock, or a Hugging Face model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.entities.medical_image import ImageAnalysisResult, ImageModality


class VisionAnalysisService(ABC):
    """
    Service contract for medical image analysis.

    Implementations encapsulate the specific SDK or API calls required
    to analyze a diagnostic image and return structured findings.
    """

    @abstractmethod
    async def analyze_image(
        self,
        image_bytes: bytes,
        modality: ImageModality,
        body_region: str = "unknown",
    ) -> ImageAnalysisResult:
        """
        Analyze a medical image and extract clinical findings.

        Args:
            image_bytes: Raw bytes of the uploaded image.
            modality: Imaging modality (X-Ray, CT, MRI, …).
            body_region: Anatomical region to focus analysis on.

        Returns:
            Structured analysis result with anomalies and description.

        Raises:
            ImageAnalysisError: If the external service fails or returns
                an unusable response.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify connectivity with the vision analysis backend."""
        ...
