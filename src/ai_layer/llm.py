"""
Groq LLM Configuration
=======================

Shared LLM configuration for the AI Decision Layer.

Provider:
    Groq

Model:
    openai/gpt-oss-120b

The API key is loaded from GROQ_API_KEY in the environment.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "openai/gpt-oss-120b"

TEMPERATURE = 0.0


# ============================================================================
# ENVIRONMENT
# ============================================================================

load_dotenv()


def get_groq_api_key() -> str:
    """Return the configured Groq API key."""

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured.\n"
            "Add it to the project .env file."
        )

    return api_key


# ============================================================================
# LLM FACTORY
# ============================================================================

def get_reasoning_llm() -> ChatGroq:
    """Create the shared Groq reasoning model."""

    return ChatGroq(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        api_key=get_groq_api_key(),
    )


if __name__ == "__main__":

    llm = get_reasoning_llm()

    response = llm.invoke(
        "Reply with exactly: Groq reasoning model is working."
    )

    print(response.content)