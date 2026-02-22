"""
Medical Image DTOs — Pydantic schemas for image upload and analysis responses.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ImageUploadResponse(BaseModel):
    """Response after uploading a medical image."""

    id: UUID
    patient_id: UUID
    filename: str
    modality: str
    body_region: str
    created_at: datetime | None

    model_config = ConfigDict(from_attributes=True)


class DetectedAnomalyResponse(BaseModel):
    """A single anomaly in the analysis response."""

    label: str
    confidence: float
    region: str
    bounding_box: list[float] | None = None


class ImageAnalysisResponse(BaseModel):
    """Response after image analysis completes."""

    image_id: UUID
    patient_id: UUID
    filename: str
    modality: str
    description: str
    anomalies: list[DetectedAnomalyResponse]
    raw_tags: list[str]
    model_version: str

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "image_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "patient_id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
            "filename": "chest_xray_001.png",
            "modality": "xray",
            "description": "Possible opacification in the right lower lobe.",
            "anomalies": [
                {
                    "label": "pulmonary_opacity",
                    "confidence": 0.87,
                    "region": "right_lower_lobe",
                    "bounding_box": [120.5, 180.2, 95.0, 110.3],
                }
            ],
            "raw_tags": ["opacity", "effusion", "normal"],
            "model_version": "azure-cv-4.0",
        }
    })
