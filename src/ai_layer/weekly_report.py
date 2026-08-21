"""
Automated Weekly Business Intelligence Report
==============================================

Builds the project's weekly business-intelligence report by calling
the production run_agent() entrypoint.

This module does NOT duplicate:
    - query routing
    - tool selection
    - forecast retrieval
    - SHAP retrieval
    - historical analysis
    - risk assessment
    - Groq reasoning

It delegates those responsibilities to the existing agent.

Output:
    reports/weekly_reports/weekly_business_intelligence_YYYYMMDD_HHMMSS.md
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.ai_layer.agent import run_agent


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

WEEKLY_REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "weekly_reports"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

WEEKLY_REPORT_QUERY = (
    "Generate a weekly business intelligence report "
    "for the overall revenue forecast. Summarize the "
    "latest overall forecast, its main TreeSHAP drivers, "
    "recent validated performance, current forecast risk, "
    "relevant project-derived business context, and "
    "evidence-based recommendations."
)


# ============================================================================
# REPORT GENERATION
# ============================================================================

def build_weekly_report() -> dict:
    """
    Generate the weekly business-intelligence report.

    Returns
    -------
    dict
        The final LangGraph result plus weekly-report metadata.
    """

    WEEKLY_REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=== AUTOMATED WEEKLY REPORT ==="
    )

    print(
        "\nQuery:"
    )

    print(
        WEEKLY_REPORT_QUERY
    )

    # ---------------------------------------------------------------
    # Reuse the production agent entrypoint.
    # ---------------------------------------------------------------

    result = run_agent(
        WEEKLY_REPORT_QUERY
    )

    # ---------------------------------------------------------------
    # Validate execution.
    # ---------------------------------------------------------------

    errors = result.get(
        "errors",
        [],
    )

    if errors:
        raise RuntimeError(
            "Weekly report agent execution failed:\n"
            + "\n".join(
                f"- {error}"
                for error in errors
            )
        )

    report = result.get(
        "report"
    )

    if not report:
        raise RuntimeError(
            "Agent completed without producing a report."
        )

    markdown = report.get(
        "markdown",
        "",
    )

    if not markdown.strip():
        raise RuntimeError(
            "Generated weekly report is empty."
        )

    # ---------------------------------------------------------------
    # Save a dedicated weekly-report artifact.
    # ---------------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = (
        WEEKLY_REPORT_DIR
        / (
            "weekly_business_intelligence_"
            f"{timestamp}.md"
        )
    )

    output_path.write_text(
        markdown,
        encoding="utf-8",
    )

    return {
        "path":
            str(
                output_path.relative_to(
                    PROJECT_ROOT
                )
            ),
        "generated_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),
        "query":
            WEEKLY_REPORT_QUERY,
        "query_context":
            result.get(
                "query_context"
            ),
        "tool_plan":
            result.get(
                "tool_plan"
            ),
        "sources":
            result.get(
                "sources",
                [],
            ),
        "markdown":
            markdown,
    }


# ============================================================================
# VALIDATION
# ============================================================================

def validate_weekly_report(
    result: dict,
) -> None:
    """Validate the generated report artifact."""

    report_path = Path(
        PROJECT_ROOT
        / result["path"]
    )

    if not report_path.is_file():
        raise FileNotFoundError(
            "Weekly report file was not created:\n"
            f"{report_path}"
        )

    if report_path.stat().st_size == 0:
        raise RuntimeError(
            "Weekly report file is empty."
        )

    required_sections = [
        "Executive Summary",
        "Forecast",
        "Forecast Drivers",
        "Historical Performance",
        "Forecast Risk",
        "Key Findings",
        "Recommendations",
        "Sources",
    ]

    content = report_path.read_text(
        encoding="utf-8"
    )

    missing_sections = [
        section
        for section in required_sections
        if section not in content
    ]

    if missing_sections:
        raise RuntimeError(
            "Weekly report is missing required "
            f"sections: {missing_sections}"
        )


# ============================================================================
# MAIN
# ============================================================================

def run_weekly_report() -> dict:
    """Generate and validate one weekly business-intelligence report."""

    result = build_weekly_report()

    validate_weekly_report(
        result
    )

    print(
        "\n=== WEEKLY REPORT COMPLETE ==="
    )

    print(
        f"Report path:\n"
        f"{result['path']}"
    )

    print(
        "\nQuery context:"
    )

    print(
        result["query_context"]
    )

    print(
        "\nTool plan:"
    )

    print(
        result["tool_plan"]
    )

    print(
        "\nSources:"
    )

    for source in result[
        "sources"
    ]:
        print(
            f"- {source}"
        )

    print(
        "\nValidation:"
    )

    print(
        "All required report sections present."
    )

    return result


if __name__ == "__main__":
    run_weekly_report()