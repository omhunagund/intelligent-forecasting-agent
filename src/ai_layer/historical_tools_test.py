"""
Smoke tests for query_historical_data().
"""

from pprint import pprint

from src.ai_layer.historical_tools import (
    query_historical_data_value,
)


def test_overall() -> None:
    print(
        "\n=== OVERALL HISTORY TEST ==="
    )

    result = query_historical_data_value(
        series_type="overall",
        series_id="overall",
        comparison_period="12 weeks",
    )

    pprint(
        result
    )


def test_category() -> None:
    print(
        "\n=== CATEGORY HISTORY TEST ==="
    )

    result = query_historical_data_value(
        series_type="category",
        series_id="Automotive",
        comparison_period="12 weeks",
    )

    pprint(
        result
    )


def test_region() -> None:
    print(
        "\n=== REGION HISTORY TEST ==="
    )

    result = query_historical_data_value(
        series_type="region",
        series_id="North",
        comparison_period="12 weeks",
    )

    pprint(
        result
    )


def test_period_alias() -> None:
    print(
        "\n=== PERIOD ALIAS TEST ==="
    )

    result = query_historical_data_value(
        series_type="region",
        series_id="North",
        comparison_period="recent 4 weeks",
    )

    print(
        f"Returned records: "
        f"{len(result['records'])}"
    )

    if len(result["records"]) != 4:
        raise AssertionError(
            "Expected 4 records."
        )


def test_invalid_series() -> None:
    print(
        "\n=== INVALID SERIES TEST ==="
    )

    try:

        query_historical_data_value(
            series_type="region",
            series_id="NotARegion",
            comparison_period="12 weeks",
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
            "Expected invalid-series ValueError."
        )


def test_invalid_period() -> None:
    print(
        "\n=== INVALID PERIOD TEST ==="
    )

    try:

        query_historical_data_value(
            series_type="overall",
            series_id="overall",
            comparison_period="next year",
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
            "Expected invalid-period ValueError."
        )


def main() -> None:

    print(
        "=== HISTORICAL TOOL TESTS ==="
    )

    test_overall()
    test_category()
    test_region()
    test_period_alias()
    test_invalid_series()
    test_invalid_period()

    print(
        "\n=== HISTORICAL TOOL TESTS PASSED ==="
    )


if __name__ == "__main__":
    main()