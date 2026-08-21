"""
Smoke test for generate_report().
"""

from pathlib import Path

from src.ai_layer.report_tools import (
    build_report,
)


def main() -> None:

    print(
        "=== REPORT TOOL TEST ==="
    )

    result = build_report(
        title=(
            "North Region Forecast "
            "Risk Report"
        ),
        executive_summary=(
            "The North regional forecast "
            "shows elevated reliability risk "
            "under the project's monitoring rules."
        ),
        forecast={
            "series_type":
                "region",
            "series_id":
                "North",
            "model":
                "xgboost",
            "forecasts": [
                {
                    "timestamp":
                        "2018-09-23",
                    "forecast_revenue":
                        3984.300048828125,
                    "lower_80":
                        1880.91878515625,
                    "upper_80":
                        7812.832832,
                    "model":
                        "xgboost",
                }
            ],
        },
        shap_explanation={
            "series_type":
                "region",
            "series_id":
                "North",
            "forecast_timestamp":
                "2018-09-23",
            "forecast_revenue":
                3984.300048828125,
            "base_value":
                4673.06591796875,
            "drivers_up": [
                {
                    "feature":
                        "lag_52",
                    "feature_value":
                        5077.04,
                    "shap_value":
                        345.4139,
                }
            ],
            "drivers_down": [
                {
                    "feature":
                        "lag_4",
                    "feature_value":
                        289.39,
                    "shap_value":
                        -760.4952,
                }
            ],
        },
        historical_data={
            "comparison_period":
                "12 weeks",
            "period_start":
                "2018-05-20",
            "period_end":
                "2018-08-05",
            "metrics": {
                "mae":
                    2805.6771573893225,
                "rmse":
                    3750.835289283438,
                "mape":
                    92.51108242965314,
            },
            "records": [],
        },
        risk_assessment={
            "status":
                "alert",
            "score":
                100.0,
            "risk_components": {
                "performance":
                    50.0,
                "drift":
                    30.0,
                "uncertainty":
                    20.0,
            },
            "performance_status":
                "alert",
            "drift_status":
                "alert",
            "interval_width":
                5931.91404684375,
            "recent_mape":
                92.51108242965314,
            "baseline_mape":
                24.415006067818755,
            "reasons": [
                (
                    "Recent model performance "
                    "is in alert status."
                ),
                (
                    "5 of 5 monitored features "
                    "have alert drift status."
                ),
            ],
            "confidence_note":
                (
                    "Forecast reliability should "
                    "be treated with caution."
                ),
        },
        findings=[
            (
                "Recent North forecast performance "
                "is materially worse than baseline."
            ),
            (
                "The forecast interval is wide "
                "relative to the forecast."
            ),
        ],
        recommendations=[
            {
                "priority":
                    "high",
                "action":
                    (
                        "Review North-region forecast "
                        "reliability before using it "
                        "for high-impact planning."
                    ),
                "rationale":
                    (
                        "Performance and drift monitoring "
                        "are both in alert status."
                    ),
            }
        ],
        sources=[
            "reports/secondary/secondary_latest_forecasts.csv",
            "reports/shap/agent_shap_explanations.json",
            "reports/secondary/secondary_predictions.parquet",
            "reports/monitoring/model_performance_report.csv",
            "reports/monitoring/data_drift_report.csv",
        ],
    )

    print(
        "\nReport generated successfully."
    )

    print(
        f"Path: {result['path']}"
    )

    print(
        "\nReport preview:"
    )

    print(
        result["markdown"][:2000]
    )

    output_path = (
        Path(
            "reports"
        )
        / "weekly_reports"
    )

    if not output_path.is_dir():
        raise AssertionError(
            "weekly_reports directory "
            "was not created."
        )

    print(
        "\n=== REPORT TOOL TEST PASSED ==="
    )


if __name__ == "__main__":
    main()