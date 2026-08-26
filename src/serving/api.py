"""
FastAPI Serving Layer
=====================

Public REST API for the Intelligent Forecasting Agent.

Endpoints required by the project blueprint:

POST /forecast
GET  /explanation
POST /agent/query
GET  /reports/latest
GET  /health
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

from src.ai_layer.agent import run_agent

from src.ai_layer.tools import (
    get_latest_forecast,
    get_shap_explanation,
)

from src.serving.schemas import (
    AgentQueryRequest,
    AgentQueryResponse,
    ExplanationResponse,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    LatestReportResponse,
)


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
)

WEEKLY_REPORTS_DIR = (
    REPORTS_DIR
    / "weekly_reports"
)

MONITORING_SUMMARY_PATH = (
    REPORTS_DIR
    / "monitoring"
    / "monitoring_summary.json"
)


# ============================================================================
# APPLICATION
# ============================================================================

app = FastAPI(
    title="Intelligent Forecasting Agent API",
    description=(
        "Serving API for ML forecasts, TreeSHAP explanations, "
        "AI-agent queries, weekly reports, and model health."
    ),
    version="1.0.0",
)


# ============================================================================
# HELPERS
# ============================================================================

def _load_monitoring_summary() -> dict:
    """Load the latest monitoring summary."""

    if not MONITORING_SUMMARY_PATH.is_file():
        return {}

    try:
        return json.loads(
            MONITORING_SUMMARY_PATH.read_text(
                encoding="utf-8"
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return {}


def _find_latest_weekly_report() -> Path | None:
    """Return the most recently modified weekly report."""

    if not WEEKLY_REPORTS_DIR.is_dir():
        return None

    reports = sorted(
        WEEKLY_REPORTS_DIR.glob(
            "weekly_business_intelligence_*.md"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not reports:
        return None

    return reports[0]


# ============================================================================
# ROOT
# ============================================================================

@app.get("/")
def root() -> dict[str, str]:
    """Basic API information."""

    return {
        "service":
            "Intelligent Forecasting Agent API",
        "version":
            "1.0.0",
        "status":
            "running",
    }


# ============================================================================
# FORECAST
# ============================================================================

@app.post(
    "/forecast",
    response_model=ForecastResponse,
)
def forecast(
    request: ForecastRequest,
) -> dict:
    """
    Get the latest ML forecast for an overall,
    category, or regional series.
    """

    try:

        result = get_latest_forecast.invoke(
            {
                "series_type":
                    request.series_type,
                "series_id":
                    request.series_id,
                "horizon":
                    request.horizon,
            }
        )

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return result


# ============================================================================
# SHAP EXPLANATION
# ============================================================================

@app.get(
    "/explanation",
    response_model=ExplanationResponse,
)
def explanation(
    series_type: str = Query(
        ...,
        description="overall, category, or region",
    ),
    series_id: str = Query(
        ...,
        description="Project series identifier",
    ),
    forecast_timestamp: str | None = Query(
        default=None,
        description=(
            "Forecast date in YYYY-MM-DD format. "
            "When omitted, the tool uses its latest available "
            "explanation."
        ),
    ),
) -> dict:

    try:

        arguments = {
            "series_type":
                series_type,
            "series_id":
                series_id,
        }

        if forecast_timestamp:
            arguments[
                "forecast_timestamp"
            ] = forecast_timestamp

        result = get_shap_explanation.invoke(
            arguments
        )

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return result


# ============================================================================
# AGENT QUERY
# ============================================================================

@app.post(
    "/agent/query",
    response_model=AgentQueryResponse,
)
def agent_query(
    request: AgentQueryRequest,
) -> dict:

    try:

        result = run_agent(
            request.query
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    return {
        "query":
            request.query,
        "query_context":
            result.get(
                "query_context",
                {},
            ),
        "tool_plan":
            result.get(
                "tool_plan",
                {},
            ),
        "synthesis":
            result.get(
                "synthesis",
                "",
            ),
        "risk_assessment":
            result.get(
                "risk_assessment"
            ),
        "report":
            result.get(
                "report"
            ),
        "sources":
            result.get(
                "sources",
                [],
            ),
    }


# ============================================================================
# LATEST REPORT
# ============================================================================

@app.get(
    "/reports/latest",
    response_model=LatestReportResponse,
)
def latest_report() -> dict:

    report_path = (
        _find_latest_weekly_report()
    )

    if report_path is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "No generated weekly business "
                "intelligence report was found."
            ),
        )

    try:

        content = report_path.read_text(
            encoding="utf-8"
        )

    except OSError as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to read latest report."
            ),
        ) from exc

    modified_time = (
        report_path.stat().st_mtime
    )

    from datetime import datetime

    generated_at = (
        datetime.fromtimestamp(
            modified_time
        ).isoformat(
            timespec="seconds"
        )
    )

    return {
        "filename":
            report_path.name,
        "path":
            str(
                report_path.relative_to(
                    PROJECT_ROOT
                )
            ),
        "generated_at":
            generated_at,
        "content":
            content,
    }


# ============================================================================
# HEALTH
# ============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> dict:
    """Return service and latest ML monitoring health."""

    monitoring = _load_monitoring_summary()

    return {
        "status": "ok",
        "service": "Intelligent Forecasting Agent API",
        "monitoring_status": monitoring.get(
            "monitoring_status"
        ),
        "drift_alerts": monitoring.get(
            "drift_alerts"
        ),
        "drift_warnings": monitoring.get(
            "drift_warnings"
        ),
        "performance_alerts": monitoring.get(
            "performance_alerts"
        ),
        "performance_warnings": monitoring.get(
            "performance_warnings"
        ),
    }