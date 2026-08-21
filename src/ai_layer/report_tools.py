"""
Business Intelligence Report Generator
======================================

Final deterministic report formatter for the AI Decision Layer.

This module does not invent new findings.

It formats already-computed evidence from:
    - forecast retrieval
    - SHAP explanations
    - historical analysis
    - business-context retrieval
    - risk assessment

into a structured Markdown business report.

No LLM logic is used here.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "weekly_reports"
)


# ============================================================================
# HELPERS
# ============================================================================

def ensure_report_directory() -> None:
    """Create the weekly-report directory."""

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def format_currency(
    value: float | int | None,
) -> str:
    """Format a numeric revenue value."""

    if value is None:
        return "N/A"

    return f"{float(value):,.2f}"


def format_percentage(
    value: float | int | None,
) -> str:
    """Format a percentage."""

    if value is None:
        return "N/A"

    return f"{float(value):.2f}%"


def safe_string(
    value: Any,
) -> str:
    """Convert a value safely to a string."""

    if value is None:
        return "N/A"

    return str(
        value
    )


def format_forecast_section(
    forecast: dict,
) -> list[str]:
    """Format forecast evidence."""

    lines = [
        "## Forecast",
        "",
        (
            f"**Series:** "
            f"{forecast.get('series_type', 'N/A')} / "
            f"{forecast.get('series_id', 'N/A')}"
        ),
        "",
        (
            f"**Model:** "
            f"{forecast.get('model', 'N/A')}"
        ),
        "",
        "| Date | Forecast Revenue | Lower 80% | Upper 80% |",
        "|---|---:|---:|---:|",
    ]

    for item in forecast.get(
        "forecasts",
        [],
    ):

        lines.append(
            "| "
            f"{item.get('timestamp', 'N/A')} | "
            f"{format_currency(item.get('forecast_revenue'))} | "
            f"{format_currency(item.get('lower_80'))} | "
            f"{format_currency(item.get('upper_80'))} |"
        )

    lines.append("")

    return lines


def format_shap_section(
    shap: dict,
) -> list[str]:
    """Format SHAP evidence."""

    lines = [
        "## Forecast Drivers",
        "",
        (
            f"Forecast date: "
            f"**{safe_string(shap.get('forecast_timestamp'))}**"
        ),
        "",
        (
            f"Forecast revenue: "
            f"**{format_currency(shap.get('forecast_revenue'))}**"
        ),
        "",
        (
            f"Base value: "
            f"**{format_currency(shap.get('base_value'))}**"
        ),
        "",
        (
            "SHAP values represent additive contributions "
            "to the model prediction in the model's output units; "
            "they are not percentages."
        ),
        "",
        "### Drivers pushing forecast upward",
        "",
    ]

    drivers_up = shap.get(
        "drivers_up",
        [],
    )

    if drivers_up:

        for rank, driver in enumerate(
            drivers_up,
            start=1,
        ):

            lines.append(
                f"{rank}. "
                f"`{driver.get('feature', 'N/A')}` "
                f"(SHAP contribution: "
                f"{format_currency(driver.get('shap_value'))})"
                " — feature value "
                f"`{driver.get('feature_value', 'N/A')}`"
            )

    else:

        lines.append(
            "_No upward drivers returned._"
        )

    lines.extend(
        [
            "",
            "### Drivers pushing forecast downward",
            "",
        ]
    )

    drivers_down = shap.get(
        "drivers_down",
        [],
    )

    if drivers_down:

        for rank, driver in enumerate(
            drivers_down,
            start=1,
        ):

            lines.append(
                f"{rank}. "
                f"`{driver.get('feature', 'N/A')}` "
                f"(SHAP contribution: "
                f"{format_currency(driver.get('shap_value'))})"
                " — feature value "
                f"`{driver.get('feature_value', 'N/A')}`"
            )

    else:

        lines.append(
            "_No downward drivers returned._"
        )

    lines.append("")

    return lines


def format_risk_section(
    risk: dict,
) -> list[str]:
    """Format risk-assessment evidence."""

    components = risk.get(
        "risk_components",
        {},
    )

    lines = [
        "## Forecast Risk",
        "",
        (
            f"**Status:** "
            f"{safe_string(risk.get('status')).upper()}"
        ),
        "",
        (
            f"**Risk score:** "
            f"{safe_string(risk.get('score'))}/100"
        ),
        "",
        "| Component | Points |",
        "|---|---:|",
        (
            f"| Performance | "
            f"{safe_string(components.get('performance'))} |"
        ),
        (
            f"| Drift | "
            f"{safe_string(components.get('drift'))} |"
        ),
        (
            f"| Uncertainty | "
            f"{safe_string(components.get('uncertainty'))} |"
        ),
        "",
        (
            f"**Performance status:** "
            f"{safe_string(risk.get('performance_status'))}"
        ),
        "",
        (
            f"**Drift status:** "
            f"{safe_string(risk.get('drift_status'))}"
        ),
        "",
        (
            f"**Interval width:** "
            f"{format_currency(risk.get('interval_width'))}"
        ),
        "",
        (
            f"**Recent MAPE:** "
            f"{format_percentage(risk.get('recent_mape'))}"
        ),
        "",
        (
            f"**Baseline MAPE:** "
            f"{format_percentage(risk.get('baseline_mape'))}"
        ),
        "",
        "### Risk reasons",
        "",
    ]

    reasons = risk.get(
        "reasons",
        [],
    )

    if reasons:

        for reason in reasons:
            lines.append(
                f"- {reason}"
            )

    else:

        lines.append(
            "- No risk reasons were returned."
        )

    lines.extend(
        [
            "",
            "### Confidence note",
            "",
            safe_string(
                risk.get(
                    "confidence_note"
                )
            ),
            "",
        ]
    )

    return lines


def format_historical_section(
    historical: dict,
) -> list[str]:
    """Format historical comparison evidence."""

    metrics = historical.get(
        "metrics",
        {},
    )

    lines = [
        "## Historical Performance",
        "",
        (
            f"Comparison period: "
            f"**{safe_string(historical.get('comparison_period'))}**"
        ),
        "",
        (
            f"Period: "
            f"**{safe_string(historical.get('period_start'))}** "
            f"to "
            f"**{safe_string(historical.get('period_end'))}**"
        ),
        "",
        "| Metric | Value |",
        "|---|---:|",
        (
            f"| MAE | "
            f"{format_currency(metrics.get('mae'))} |"
        ),
        (
            f"| RMSE | "
            f"{format_currency(metrics.get('rmse'))} |"
        ),
        (
            f"| MAPE | "
            f"{format_percentage(metrics.get('mape'))} |"
        ),
        "",
    ]

    records = historical.get(
        "records",
        [],
    )

    if records:

        lines.extend(
            [
                "### Historical observations",
                "",
                "| Date | Actual Revenue | Forecast Revenue | Absolute Error |",
                "|---|---:|---:|---:|",
            ]
        )

        for record in records:

            lines.append(
                "| "
                f"{record.get('timestamp', 'N/A')} | "
                f"{format_currency(record.get('actual_revenue'))} | "
                f"{format_currency(record.get('forecast_revenue'))} | "
                f"{format_currency(record.get('absolute_error'))} |"
            )

        lines.append("")

    return lines


def format_context_section(
    business_context: dict,
) -> list[str]:
    """Format retrieved project-derived context."""

    lines = [
        "## Retrieved Business Context",
        "",
    ]

    results = business_context.get(
        "results",
        [],
    )

    if not results:

        lines.append(
            "_No business-context results were retrieved._"
        )

        lines.append("")

        return lines

    for result in results:

        lines.extend(
            [
                (
                    f"### Source {result.get('rank', 'N/A')}: "
                    f"`{result.get('source', 'unknown')}`"
                ),
                "",
                (
                    f"**Section:** "
                    f"{result.get('section', 'unknown')}"
                ),
                "",
                result.get(
                    "text",
                    "",
                ),
                "",
            ]
        )

    return lines


# ============================================================================
# MAIN REPORT BUILDER
# ============================================================================

def build_report(
    title: str,
    executive_summary: str,
    forecast: dict | None = None,
    shap_explanation: dict | None = None,
    historical_data: dict | None = None,
    business_context: dict | None = None,
    risk_assessment: dict | None = None,
    findings: list[str] | None = None,
    recommendations: list[dict] | None = None,
    sources: list[str] | None = None,
) -> dict:
    """
    Build and save a structured business-intelligence report.

    No new analytical claims are generated here. All findings and
    recommendations supplied to this function must already have
    been produced by the reasoning layer.
    """

    if not title.strip():
        raise ValueError(
            "title must not be empty."
        )

    if not executive_summary.strip():
        raise ValueError(
            "executive_summary must not be empty."
        )

    ensure_report_directory()

    generated_at = datetime.now().isoformat(
        timespec="seconds"
    )

    lines = [
        f"# {title}",
        "",
        f"**Generated:** {generated_at}",
        "",
        "## Executive Summary",
        "",
        executive_summary.strip(),
        "",
    ]

    # ---------------------------------------------------------------
    # Forecast
    # ---------------------------------------------------------------

    if forecast is not None:

        lines.extend(
            format_forecast_section(
                forecast
            )
        )

    # ---------------------------------------------------------------
    # SHAP
    # ---------------------------------------------------------------

    if shap_explanation is not None:

        lines.extend(
            format_shap_section(
                shap_explanation
            )
        )

    # ---------------------------------------------------------------
    # Historical
    # ---------------------------------------------------------------

    if historical_data is not None:

        lines.extend(
            format_historical_section(
                historical_data
            )
        )

    # ---------------------------------------------------------------
    # Risk
    # ---------------------------------------------------------------

    if risk_assessment is not None:

        lines.extend(
            format_risk_section(
                risk_assessment
            )
        )

    # ---------------------------------------------------------------
    # Business context
    # ---------------------------------------------------------------

    if business_context is not None:

        lines.extend(
            format_context_section(
                business_context
            )
        )

    # ---------------------------------------------------------------
    # Findings
    # ---------------------------------------------------------------

    lines.extend(
        [
            "## Key Findings",
            "",
        ]
    )

    if findings:

        for finding in findings:

            lines.append(
                f"- {finding}"
            )

    else:

        lines.append(
            "_No additional findings were supplied._"
        )

    lines.append("")

    # ---------------------------------------------------------------
    # Recommendations
    # ---------------------------------------------------------------

    lines.extend(
        [
            "## Recommendations",
            "",
        ]
    )

    if recommendations:

        for index, recommendation in enumerate(
            recommendations,
            start=1,
        ):

            priority = recommendation.get(
                "priority",
                "medium",
            )

            action = recommendation.get(
                "action",
                "",
            )

            rationale = recommendation.get(
                "rationale",
                "",
            )

            lines.append(
                f"{index}. "
                f"**[{str(priority).upper()}]** "
                f"{action}"
            )

            if rationale:

                lines.append(
                    f"   - Rationale: {rationale}"
                )

    else:

        lines.append(
            "_No recommendations were supplied._"
        )

    lines.append("")

    # ---------------------------------------------------------------
    # Sources
    # ---------------------------------------------------------------

    lines.extend(
        [
            "## Sources",
            "",
        ]
    )

    source_list = (
        sources
        if sources
        else []
    )

    if source_list:

        for source in source_list:

            lines.append(
                f"- `{source}`"
            )

    else:

        lines.append(
            "_No explicit sources supplied._"
        )

    lines.append("")

    # ---------------------------------------------------------------
    # Transparency note
    # ---------------------------------------------------------------

    lines.extend(
        [
            "---",
            "",
            (
                "This report contains project-derived forecast, "
                "explainability, historical validation, monitoring, "
                "and retrieved-context outputs."
            ),
            "",
            (
                "Production forecast outputs are generated by the "
                "selected production XGBoost model. Historical "
                "validation evidence may come from the project's "
                "walk-forward validation artifacts and should not be "
                "interpreted as production-model performance unless "
                "explicitly identified as such."
            ),
            "",
            (
                "Project-defined risk thresholds and recommendations "
                "are system design choices, not external industry "
                "standards."
            ),
            "",
        ]
    )

    markdown = "\n".join(
        lines
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    safe_title = (
        title.lower()
        .replace(
            " ",
            "_",
        )
        .replace(
            "/",
            "_",
        )
        .replace(
            "\\",
            "_",
        )
    )

    output_path = (
        REPORT_DIR
        / f"{safe_title}_{timestamp}.md"
    )

    output_path.write_text(
        markdown,
        encoding="utf-8",
    )

    return {
        "title":
            title,
        "generated_at":
            generated_at,
        "markdown":
            markdown,
        "path":
            str(
                output_path.relative_to(
                    PROJECT_ROOT
                )
            ),
        "sources":
            source_list,
    }