"""
Smoke tests for assess_forecast_risk().
"""

from pprint import pprint

from src.ai_layer.risk_tools import (
    assess_forecast_risk_data,
)


def test_overall() -> None:
    print(
        "\n=== OVERALL RISK TEST ==="
    )

    result = assess_forecast_risk_data(
        series_type="overall",
        series_id="overall",
        forecast_revenue=271394.25,
        lower_80=161562.245375,
        upper_80=320540.631656,
    )

    pprint(
        result
    )


def test_category() -> None:
    print(
        "\n=== CATEGORY RISK TEST ==="
    )

    result = assess_forecast_risk_data(
        series_type="category",
        series_id="Automotive",
        forecast_revenue=14574.654296875,
        lower_80=12341.486068359374,
        upper_80=20589.2669921875,
    )

    pprint(
        result
    )


def test_region() -> None:
    print(
        "\n=== REGION RISK TEST ==="
    )

    result = assess_forecast_risk_data(
        series_type="region",
        series_id="North",
        forecast_revenue=3984.300048828125,
        lower_80=1880.91878515625,
        upper_80=7812.832832,
    )

    pprint(
        result
    )


def test_invalid_interval() -> None:
    print(
        "\n=== INVALID INTERVAL TEST ==="
    )

    try:

        assess_forecast_risk_data(
            series_type="overall",
            series_id="overall",
            forecast_revenue=100000.0,
            lower_80=120000.0,
            upper_80=90000.0,
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
            "Expected invalid-interval ValueError."
        )


def main() -> None:

    print(
        "=== RISK TOOL TESTS ==="
    )

    test_overall()
    test_category()
    test_region()
    test_invalid_interval()

    print(
        "\n=== RISK TOOL TESTS PASSED ==="
    )


if __name__ == "__main__":
    main()