"""
Azure OpenAI Client — Wrapper for GPT model access.

Provides a configured ``AzureChatOpenAI`` instance from LangChain
that connects to an Azure OpenAI deployment for clinical reasoning.
"""

from __future__ import annotations

import logging

from langchain_openai import AzureChatOpenAI

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def get_azure_llm() -> AzureChatOpenAI:
    """
    Build a LangChain-compatible Azure OpenAI chat model.

    Returns:
        Configured ``AzureChatOpenAI`` instance ready for chain/agent use.
    """
    settings = get_settings()

    return AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_deployment_name,
        temperature=0.2,  # Low temperature for clinical precision
        max_tokens=2048,
    )
