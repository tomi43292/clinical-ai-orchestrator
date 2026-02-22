"""
Medical Image Entity — Domain representation of a diagnostic image study.

Captures metadata about the image, the modality (X-Ray, CT, MRI), the
body region, and the structured results returned by the Computer Vision
analysis service.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4


class ImageModality(str, Enum):
    """Supported diagnostic imaging modalities."""

    XRAY = "xray"
    CT = "ct"
    MRI = "mri"
    ULTRASOUND = "ultrasound"
    OTHER = "other"


class BodyRegion(str, Enum):
    """Anatomical regions for image classification."""

    CHEST = "chest"
    ABDOMEN = "abdomen"
    HEAD = "head"
    SPINE = "spine"
    EXTREMITIES = "extremities"
    PELVIS = "pelvis"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class DetectedAnomaly:
    """
    A single anomaly detected by the vision model.

    Attributes:
        label: Descriptive label (e.g., "opacity", "fracture").
        confidence: Model confidence score between 0 and 1.
        region: Textual description of the affected area.
        bounding_box: Optional [x, y, width, height] coordinates.
    """

    label: str
    confidence: float
    region: str
    bounding_box: list[float] | None = None


@dataclass(frozen=True, slots=True)
class ImageAnalysisResult:
    """
    Aggregated output from the Computer Vision analysis.

    Attributes:
        anomalies: List of detected anomalies.
        description: Natural-language summary of findings.
        raw_tags: Raw tags returned by the vision model.
        model_version: Identifier of the vision model/version used.
    """

    anomalies: list[DetectedAnomaly] = field(default_factory=list)
    description: str = ""
    raw_tags: list[str] = field(default_factory=list)
    model_version: str = ""


@dataclass(frozen=True, slots=True)
class MedicalImage:
    """
    Domain entity for a medical image submitted for analysis.

    Attributes:
        id: Unique identifier (UUIDv4).
        patient_id: Foreign key to the owning Patient.
        filename: Original filename of the uploaded image.
        content_type: MIME type (e.g., "image/png").
        storage_path: Path or URL where the image bytes are stored.
        modality: Imaging modality (X-Ray, CT, MRI, …).
        body_region: Anatomical region depicted.
        analysis_result: Structured CV analysis output (populated after analysis).
        created_at: Upload timestamp.
    """

    id: UUID = field(default_factory=uuid4)
    patient_id: UUID | None = None
    filename: str = ""
    content_type: str = ""
    storage_path: str = ""
    modality: ImageModality = ImageModality.OTHER
    body_region: BodyRegion = BodyRegion.OTHER
    analysis_result: ImageAnalysisResult | None = None
    created_at: datetime | None = None
