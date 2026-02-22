"""
Azure Machine Learning Client — Model registry and inference operations.

Provides connectivity to Azure ML workspaces for model versioning,
deployment status checks, and managed endpoint invocations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Metadata about a registered ML model."""

    name: str
    version: str
    description: str
    tags: dict[str, str]


class AzureMLClient:
    """
    Client for Azure ML workspace operations.

    Supports model listing, health checks, and managed endpoint
    invocations.  In ``mock`` mode, methods return synthetic data.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._subscription_id = settings.azure_ml_subscription_id
        self._resource_group = settings.azure_ml_resource_group
        self._workspace_name = settings.azure_ml_workspace_name
        self._ml_client = None

    def _get_ml_client(self):
        """Lazily initialize the Azure ML client."""
        if self._ml_client is None:
            try:
                from azure.ai.ml import MLClient
                from azure.identity import DefaultAzureCredential

                self._ml_client = MLClient(
                    credential=DefaultAzureCredential(),
                    subscription_id=self._subscription_id,
                    resource_group_name=self._resource_group,
                    workspace_name=self._workspace_name,
                )
            except Exception as exc:
                logger.warning("Azure ML client init failed: %s", exc)
        return self._ml_client

    async def list_models(self) -> list[ModelInfo]:
        """List all registered models in the workspace."""
        client = self._get_ml_client()
        if client is None:
            return []

        try:
            models = client.models.list()
            return [
                ModelInfo(
                    name=m.name,
                    version=m.version,
                    description=m.description or "",
                    tags=m.tags or {},
                )
                for m in models
            ]
        except Exception as exc:
            logger.error("Failed to list ML models: %s", exc)
            return []

    async def get_model_info(self, name: str, version: str = "latest") -> ModelInfo | None:
        """Retrieve metadata for a specific model version."""
        client = self._get_ml_client()
        if client is None:
            return None

        try:
            model = client.models.get(name=name, version=version)
            return ModelInfo(
                name=model.name,
                version=model.version,
                description=model.description or "",
                tags=model.tags or {},
            )
        except Exception as exc:
            logger.error("Failed to get model '%s': %s", name, exc)
            return None

    async def health_check(self) -> bool:
        """Verify connectivity to Azure ML workspace."""
        client = self._get_ml_client()
        return client is not None
