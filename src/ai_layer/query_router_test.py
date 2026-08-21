"""
Smoke tests for natural-language query routing.
"""

from pprint import pprint

from src.ai_layer.query_router import (
    route_user_query,
)


TEST_QUERIES = [
    "Why is North revenue forecast risky?",
    "What is the next 4 week revenue forecast for Automotive?",
    "Explain what is driving the overall revenue forecast.",
    "How has Beauty & Health forecast performance been recently?",
    "Give me a report on the South region.",
]


def main() -> None:

    print(
        "=== QUERY ROUTER TESTS ==="
    )

    for query in TEST_QUERIES:

        print(
            "\nQuery:"
        )

        print(
            query
        )

        result = route_user_query(
            query
        )

        pprint(
            result
        )

        if not result.get(
            "series_type"
        ):
            raise AssertionError(
                "Router did not return series_type."
            )

        if not result.get(
            "series_id"
        ):
            raise AssertionError(
                "Router did not return series_id."
            )

    print(
        "\n=== QUERY ROUTER TESTS PASSED ==="
    )


if __name__ == "__main__":
    main()