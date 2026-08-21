"""
Smoke test for retrieve_business_context().
"""

from pprint import pprint

from src.ai_layer.rag_tools import (
    retrieve_business_context_data,
)


def run_test(
    query: str,
) -> None:

    print(
        "\n=== RAG QUERY ==="
    )

    print(
        query
    )

    result = retrieve_business_context_data(
        query=query,
        top_k=3,
    )

    print(
        "\nRetrieved sources:"
    )

    for source in result["sources"]:
        print(
            f"- {source}"
        )

    print(
        "\nTop results:"
    )

    for item in result["results"]:

        print(
            f"\nRank: {item['rank']}"
        )

        print(
            f"Source: {item['source']}"
        )

        print(
            f"Section: {item['section']}"
        )

        print(
            f"Distance: {item['distance']}"
        )

        preview = (
            item["text"]
            .replace(
                "\n",
                " ",
            )
        )

        if len(preview) > 300:
            preview = (
                preview[:300]
                + "..."
            )

        print(
            f"Text: {preview}"
        )


def test_invalid_query() -> None:

    print(
        "\n=== INVALID QUERY TEST ==="
    )

    try:

        retrieve_business_context_data(
            query="",
            top_k=3,
        )

    except ValueError as exc:

        print(
            "Expected error:"
        )

        print(
            exc
        )

    else:

        raise AssertionError(
            "Expected empty-query ValueError."
        )


def main() -> None:

    print(
        "=== RAG TOOL TESTS ==="
    )

    run_test(
        "What features are driving "
        "the overall revenue forecast?"
    )

    run_test(
        "Which categories have strong "
        "or weak forecast performance?"
    )

    run_test(
        "What is the current forecast "
        "monitoring status?"
    )

    test_invalid_query()

    print(
        "\n=== RAG TOOL TESTS PASSED ==="
    )


if __name__ == "__main__":
    main()