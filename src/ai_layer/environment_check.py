"""
AI-layer environment validation.
"""

from __future__ import annotations

import os
from importlib.metadata import version, PackageNotFoundError

from dotenv import load_dotenv


REQUIRED_PACKAGES = [
    "langgraph",
    "langchain-core",
    "langchain-groq",
    "chromadb",
    "sentence-transformers",
    "python-dotenv",
    "tabulate",
]


def check_package(
    package_name: str,
) -> str:

    try:
        return version(
            package_name
        )
    except PackageNotFoundError:
        return "NOT INSTALLED"


def main() -> None:

    print(
        "=== AI LAYER ENVIRONMENT CHECK ==="
    )

    load_dotenv()

    print(
        "\nRequired packages:"
    )

    for package in REQUIRED_PACKAGES:

        installed_version = check_package(
            package
        )

        print(
            f"- {package}: "
            f"{installed_version}"
        )

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    print(
        "\nGroq API key:"
    )

    print(
        "configured"
        if api_key
        else "MISSING"
    )

    print(
        "\n=== ENVIRONMENT CHECK COMPLETE ==="
    )


if __name__ == "__main__":
    main()