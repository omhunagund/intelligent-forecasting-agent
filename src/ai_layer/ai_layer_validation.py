"""
Final AI-layer validation.

Validates the complete AI Decision Layer without duplicating any
business logic.
"""

from __future__ import annotations

from pathlib import Path

from src.ai_layer.agent import run_agent
from src.ai_layer.rag_tools import (
    retrieve_business_context_data,
)
from src.ai_layer.forecast_tools import (
    get_latest_forecast_data,
)
from src.ai_layer.shap_tools import (
    get_shap_explanation_data,
)
from src.ai_layer.historical_tools import (
    query_historical_data_value,
)
from src.ai_layer.risk_tools import (
    assess_forecast_risk_data,
)
from src.ai_layer.weekly_report import (
    run_weekly_report,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)


def validate_forecast() -> None:

    result = get_latest_forecast_data(
        "overall",
        "overall",
        4,
    )

    assert (
        result["model"]
        == "xgboost"
    )

    assert (
        len(result["forecasts"])
        == 4
    )


def validate_shap() -> None:

    result = get_shap_explanation_data(
        "overall",
        "overall",
    )

    assert (
        result["series_type"]
        == "overall"
    )

    assert (
        "drivers_up"
        in result
    )

    assert (
        "drivers_down"
        in result
    )


def validate_rag() -> None:

    result = (
        retrieve_business_context_data(
            "What is the current forecast monitoring status?",
            top_k=3,
        )
    )

    assert result[
        "results"
    ]


def validate_historical() -> None:

    result = (
        query_historical_data_value(
            "overall",
            "overall",
            "12 weeks",
        )
    )

    assert len(
        result["records"]
    ) == 12

    assert (
        result["metrics"]["mape"]
        is not None
    )


def validate_risk() -> None:

    result = (
        assess_forecast_risk_data(
            series_type="region",
            series_id="North",
            forecast_revenue=3984.300048828125,
            lower_80=1880.91878515625,
            upper_80=7812.832832,
        )
    )

    assert (
        result["status"]
        in {
            "stable",
            "warning",
            "alert",
        }
    )

    assert (
        0 <= result["score"] <= 100
    )


def validate_agent() -> None:

    result = run_agent(
        "Why is North revenue forecast risky?"
    )

    assert (
        result.get(
            "errors",
            [],
        )
        == []
    )

    assert result.get(
        "query_context"
    )

    assert result.get(
        "tool_plan"
    )

    assert result.get(
        "synthesis"
    )

    assert result.get(
        "report"
    )


def validate_weekly_report() -> None:

    result = run_weekly_report()

    report_path = (
        PROJECT_ROOT
        / result["path"]
    )

    assert report_path.is_file()

    assert report_path.stat().st_size > 0


def main() -> None:

    print(
        "=== FINAL AI LAYER VALIDATION ==="
    )

    tests = [
        (
            "Production forecast",
            validate_forecast,
        ),
        (
            "SHAP retrieval",
            validate_shap,
        ),
        (
            "RAG retrieval",
            validate_rag,
        ),
        (
            "Historical data",
            validate_historical,
        ),
        (
            "Risk assessment",
            validate_risk,
        ),
        (
            "LangGraph agent",
            validate_agent,
        ),
        (
            "Weekly report",
            validate_weekly_report,
        ),
    ]

    for name, test in tests:

        print(
            f"\n{name}..."
        )

        test()

        print(
            f"{name}: PASS"
        )

    print(
        "\n=== FINAL AI LAYER VALIDATION PASSED ==="
    )


if __name__ == "__main__":
    main()