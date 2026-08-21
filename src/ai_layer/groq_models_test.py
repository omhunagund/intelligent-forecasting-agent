from __future__ import annotations

import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


def main() -> None:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured."
        )

    client = Groq(
        api_key=api_key
    )

    models = client.models.list()

    print("=== AVAILABLE GROQ MODELS ===")

    for model in models.data:
        print(model.id)


if __name__ == "__main__":
    main()