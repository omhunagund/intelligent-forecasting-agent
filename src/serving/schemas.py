"""
FastAPI request and response schemas.

The schemas define the public contract of the Serving Layer.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ============================================================================
# FORECAST
# ============================================================================

class ForecastRequest(BaseModel):
    """Request for a forecast."""

    series_type: str = Field(
        ...,
        description="overall, category, or region",
    )

    series_id: str = Field(
        ...,
        description="Project series identifier",
    )

    horizon: int = Field(
        default=4,
        ge=1,
        le=52,
        description="Number of future weekly forecasts",
    )


class ForecastResponse(BaseModel):
    """Forecast response."""

    forecast_horizon: int
    forecasts: list[dict[str, Any]]
    model: str
    series_id: str
    series_type: str
    source: str


# ============================================================================
# EXPLANATION
# ============================================================================

class ExplanationResponse(BaseModel):
    """TreeSHAP explanation response."""

    series_type: str
    series_id: str
    forecast_timestamp: str
    forecast_revenue: float
    base_value: float
    drivers_up: list[dict[str, Any]]
    drivers_down: list[dict[str, Any]]
    source: str


# ============================================================================
# AGENT
# ============================================================================

class AgentQueryRequest(BaseModel):
    """Natural-language query submitted to the AI agent."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural-language business question",
    )


class AgentQueryResponse(BaseModel):
    """AI agent response."""

    query: str
    query_context: dict[str, Any]
    tool_plan: dict[str, bool]
    synthesis: str
    risk_assessment: dict[str, Any] | None = None
    report: dict[str, Any] | None = None
    sources: list[str] = []


# ============================================================================
# REPORT
# ============================================================================

class LatestReportResponse(BaseModel):
    """Latest generated weekly report."""

    filename: str
    path: str
    generated_at: str | None = None
    content: str


# ============================================================================
# HEALTH
# ============================================================================

class HealthResponse(BaseModel):
    """Serving and model health information."""

    status: str
    service: str
    monitoring_status: str | None = None
    drift_alerts: int | None = None
    drift_warnings: int | None = None
    performance_alerts: int | None = None
    performance_warnings: int | None = None