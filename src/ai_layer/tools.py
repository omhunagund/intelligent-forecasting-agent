"""
AI Agent Tool Contracts
=======================

Defines the six tools required by the project specification.

Tool implementations are deliberately separated from the LangGraph
orchestration layer.

Required tools:
    1. get_latest_forecast()
    2. get_shap_explanation()
    3. retrieve_business_context()
    4. query_historical_data()
    5. assess_forecast_risk()
    6. generate_report()

At this stage these functions validate their inputs and raise
NotImplementedError. Their actual implementations will be added
one by one in the next AI-layer steps.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool


SeriesType = Literal[
    "overall",
    "category",
    "region",
]


# ============================================================================
# TOOL 1 — LATEST FORECAST
# ============================================================================

from src.ai_layer.forecast_tools import (
    get_latest_forecast_data,
)

@tool
def get_latest_forecast(
    series_type: SeriesType,
    series_id: str,
    horizon: int = 4,
) -> dict:
    """
    Retrieve the latest saved ML forecast for a specific
    overall, category, or regional series.
    """

    if horizon < 1:
        raise ValueError(
            "horizon must be at least 1."
        )

    if (
        series_type == "overall"
        and series_id != "overall"
    ):
        raise ValueError(
            "For series_type='overall', "
            "series_id must be 'overall'."
        )

    return get_latest_forecast_data(
        series_type=series_type,
        series_id=series_id,
        horizon=horizon,
    )


# ============================================================================
# TOOL 2 — SHAP EXPLANATION
# ============================================================================

from src.ai_layer.shap_tools import (
    get_shap_explanation_data,
)

@tool
def get_shap_explanation(
    series_type: SeriesType,
    series_id: str,
    forecast_timestamp: str | None = None,
) -> dict:
    """
    Retrieve the TreeSHAP explanation for a production XGBoost
    forecast.
    """

    if (
        series_type == "overall"
        and series_id != "overall"
    ):
        raise ValueError(
            "For series_type='overall', "
            "series_id must be 'overall'."
        )

    return get_shap_explanation_data(
        series_type=series_type,
        series_id=series_id,
        forecast_timestamp=forecast_timestamp,
    )


# ============================================================================
# TOOL 3 — BUSINESS CONTEXT / RAG
# ============================================================================

from src.ai_layer.rag_tools import (
    retrieve_business_context_data,
)

@tool
def retrieve_business_context(
    query: str,
    top_k: int = 5,
) -> dict:
    """
    Perform semantic retrieval over the project-derived
    ChromaDB knowledge base.
    """

    query = query.strip()

    if not query:
        raise ValueError(
            "query must not be empty."
        )

    if top_k < 1:
        raise ValueError(
            "top_k must be at least 1."
        )

    return retrieve_business_context_data(
        query=query,
        top_k=top_k,
    )


# ============================================================================
# TOOL 4 — HISTORICAL DATA
# ============================================================================

from src.ai_layer.historical_tools import (
    query_historical_data_value,
)

@tool
def query_historical_data(
    series_type: SeriesType,
    series_id: str,
    comparison_period: str,
) -> dict:
    """
    Query project-derived historical revenue and forecast outcomes.
    """

    if not comparison_period.strip():
        raise ValueError(
            "comparison_period must not be empty."
        )

    if (
        series_type == "overall"
        and series_id != "overall"
    ):
        raise ValueError(
            "For series_type='overall', "
            "series_id must be 'overall'."
        )

    return query_historical_data_value(
        series_type=series_type,
        series_id=series_id,
        comparison_period=comparison_period,
    )


# ============================================================================
# TOOL 5 — FORECAST RISK
# ============================================================================

from src.ai_layer.risk_tools import (
    assess_forecast_risk_data,
)

@tool
def assess_forecast_risk(
    series_type: SeriesType,
    series_id: str,
    forecast_revenue: float,
    lower_80: float,
    upper_80: float,
) -> dict:
    """
    Assess forecast reliability using project monitoring signals,
    uncertainty, and recent model performance.
    """

    if lower_80 > upper_80:
        raise ValueError(
            "lower_80 cannot exceed upper_80."
        )

    if (
        series_type == "overall"
        and series_id != "overall"
    ):
        raise ValueError(
            "For series_type='overall', "
            "series_id must be 'overall'."
        )

    return assess_forecast_risk_data(
        series_type=series_type,
        series_id=series_id,
        forecast_revenue=forecast_revenue,
        lower_80=lower_80,
        upper_80=upper_80,
    )


# ============================================================================
# TOOL 6 — REPORT GENERATION
# ============================================================================

from src.ai_layer.report_tools import (
    build_report,
)

@tool
def generate_report(
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
    Generate a structured business-intelligence report from
    already-computed project evidence.

    The tool does not generate new analytical claims. It passes
    verified forecast, SHAP, historical, business-context, risk,
    findings, recommendations, and source evidence to the
    deterministic report builder.
    """

    if not title.strip():
        raise ValueError(
            "title must not be empty."
        )

    if not executive_summary.strip():
        raise ValueError(
            "executive_summary must not be empty."
        )

    return build_report(
        title=title,
        executive_summary=executive_summary,
        forecast=forecast,
        shap_explanation=shap_explanation,
        historical_data=historical_data,
        business_context=business_context,
        risk_assessment=risk_assessment,
        findings=findings,
        recommendations=recommendations,
        sources=sources,
    )


# ============================================================================
# TOOL REGISTRY
# ============================================================================

AGENT_TOOLS = [
    get_latest_forecast,
    get_shap_explanation,
    retrieve_business_context,
    query_historical_data,
    assess_forecast_risk,
    generate_report,
]