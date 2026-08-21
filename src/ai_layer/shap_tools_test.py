"""
Smoke test for get_shap_explanation().
"""

from pprint import pprint

from src.ai_layer.shap_tools import (
    get_shap_explanation_data,
)


def test_overall() -> None:
    print(
        "\n=== OVERALL SHAP TEST ==="
    )

    result = get_shap_explanation_data(
        series_type="overall",
        series_id="overall",
    )

    pprint(
        result
    )


def test_category() -> None:
    print(
        "\n=== CATEGORY SHAP TEST ==="
    )

    result = get_shap_explanation_data(
        series_type="category",
        series_id="Automotive",
    )

    pprint(
        result
    )


def test_region() -> None:
    print(
        "\n=== REGION SHAP TEST ==="
    )

    result = get_shap_explanation_data(
        series_type="region",
        series_id="North",
    )

    pprint(
        result
    )


def test_exact_date() -> None:
    print(
        "\n=== EXACT-DATE SHAP TEST ==="
    )

    result = get_shap_explanation_data(
        series_type="overall",
        series_id="overall",
        forecast_timestamp="2018-09-02",
    )

    pprint(
        result
    )


def test_invalid_date() -> None:
    print(
        "\n=== INVALID-DATE SHAP TEST ==="
    )

    try:

        get_shap_explanation_data(
            series_type="overall",
            series_id="overall",
            forecast_timestamp="2030-01-01",
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
            "Expected invalid-date ValueError."
        )


def main() -> None:
    print(
        "=== SHAP TOOL TESTS ==="
    )

    test_overall()
    test_category()
    test_region()
    test_exact_date()
    test_invalid_date()

    print(
        "\n=== SHAP TOOL TESTS PASSED ==="
    )


if __name__ == "__main__":
    main()