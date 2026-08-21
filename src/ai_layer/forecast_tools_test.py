"""
Smoke test for get_latest_forecast().
"""

from pprint import pprint

from src.ai_layer.forecast_tools import (
    get_latest_forecast_data,
)


def test_overall() -> None:
    print(
        "\n=== OVERALL FORECAST TEST ==="
    )

    result = get_latest_forecast_data(
        series_type="overall",
        series_id="overall",
        horizon=4,
    )

    pprint(
        result
    )


def test_category() -> None:
    print(
        "\n=== CATEGORY FORECAST TEST ==="
    )

    result = get_latest_forecast_data(
        series_type="category",
        series_id="Automotive",
        horizon=4,
    )

    pprint(
        result
    )


def test_region() -> None:
    print(
        "\n=== REGION FORECAST TEST ==="
    )

    result = get_latest_forecast_data(
        series_type="region",
        series_id="North",
        horizon=4,
    )

    pprint(
        result
    )


def test_invalid_series() -> None:
    print(
        "\n=== INVALID SERIES TEST ==="
    )

    try:

        get_latest_forecast_data(
            series_type="region",
            series_id="NotARegion",
            horizon=4,
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


def main() -> None:
    print(
        "=== FORECAST TOOL TESTS ==="
    )

    test_overall()
    test_category()
    test_region()
    test_invalid_series()

    print(
        "\n=== FORECAST TOOL TESTS PASSED ==="
    )


if __name__ == "__main__":
    main()