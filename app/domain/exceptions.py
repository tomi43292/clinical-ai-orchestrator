"""
Domain Exceptions — Custom error hierarchy for the clinical domain.

Each exception maps to a specific failure scenario, enabling the API layer
to translate domain errors into appropriate HTTP status codes.
"""


class DomainError(Exception):
    """Base exception for all domain-level errors."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class PatientNotFoundError(DomainError):
    """Raised when a patient lookup yields no results."""

    def __init__(self, patient_id: str) -> None:
        super().__init__(
            message=f"Patient with id '{patient_id}' was not found.",
            details={"patient_id": patient_id},
        )


class MedicalImageNotFoundError(DomainError):
    """Raised when a medical image record cannot be located."""

    def __init__(self, image_id: str) -> None:
        super().__init__(
            message=f"Medical image with id '{image_id}' was not found.",
            details={"image_id": image_id},
        )


class ImageAnalysisError(DomainError):
    """Raised when the computer vision service fails to analyze an image."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            message=f"Image analysis failed: {reason}",
            details={"reason": reason},
        )


class TriageGenerationError(DomainError):
    """Raised when the LLM-based triage pipeline cannot produce a report."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            message=f"Triage report generation failed: {reason}",
            details={"reason": reason},
        )


class ClinicalNoteNotFoundError(DomainError):
    """Raised when a clinical note cannot be found."""

    def __init__(self, note_id: str) -> None:
        super().__init__(
            message=f"Clinical note with id '{note_id}' was not found.",
            details={"note_id": note_id},
        )


class SearchIndexError(DomainError):
    """Raised when indexing or querying the search service fails."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            message=f"Clinical search operation failed: {reason}",
            details={"reason": reason},
        )


class InsufficientDataError(DomainError):
    """Raised when a triage cannot be generated due to missing data."""

    def __init__(self, missing_items: list[str]) -> None:
        super().__init__(
            message=f"Insufficient data to generate triage: {', '.join(missing_items)}",
            details={"missing_items": missing_items},
        )
