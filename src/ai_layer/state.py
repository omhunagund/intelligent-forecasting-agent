"""
Agent State and Data Contracts
==============================

Defines the shared state passed between LangGraph nodes and the
structured contracts used by the AI Decision Layer.

The state is intentionally explicit so each node knows:
    - what the user asked
    - which forecasting scope is relevant
    - which ML outputs have been retrieved
    - what RAG context was found
    - what risk assessment was produced
    - what recommendations were generated
    - what final report was produced

No LLM logic is implemented here.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated


# ============================================================================
# COMMON TYPES
# ============================================================================

SeriesType = Literal[
    "overall",
    "category",
    "region",
]


RequestType = Literal[
    "forecast",
    "explanation",
    "comparison",
    "risk",
    "report",
    "general_business_question",
]


RiskStatus = Literal[
    "stable",
    "warning",
    "alert",
]


# ============================================================================
# USER REQUEST
# ============================================================================

class QueryContext(TypedDict, total=False):
    """
    Structured interpretation of the user's request.
    """

    request_type: RequestType

    series_type: SeriesType

    series_id: str

    start_date: str

    end_date: str

    forecast_horizon: int

    comparison_period: str


# ============================================================================
# TOOL OUTPUT CONTRACTS
# ============================================================================

class ForecastRecord(TypedDict, total=False):
    """One forecast record."""

    series_type: SeriesType

    series_id: str

    timestamp: str

    forecast_revenue: float

    lower_80: float

    upper_80: float

    model: str


class ShapDriver(TypedDict, total=False):
    """One SHAP feature contribution."""

    feature: str

    feature_value: float

    shap_value: float

    direction: Literal[
        "up",
        "down",
        "neutral",
    ]


class ShapExplanation(TypedDict, total=False):
    """Structured SHAP explanation."""

    series_type: SeriesType

    series_id: str

    forecast_timestamp: str

    forecast_revenue: float

    base_value: float

    drivers_up: list[ShapDriver]

    drivers_down: list[ShapDriver]


class BusinessContextResult(TypedDict, total=False):
    """RAG retrieval result."""

    query: str

    documents: list[dict[str, Any]]

    sources: list[str]


class HistoricalRecord(TypedDict, total=False):
    """Historical comparison record."""

    timestamp: str

    actual_revenue: float

    forecast_revenue: float

    absolute_error: float


class HistoricalDataResult(TypedDict, total=False):
    """Historical data query result."""

    series_type: SeriesType

    series_id: str

    comparison_period: str

    records: list[HistoricalRecord]


class RiskAssessment(TypedDict, total=False):
    """Structured forecast-risk assessment."""

    status: RiskStatus

    score: float

    reasons: list[str]

    drift_status: str

    performance_status: str

    interval_width: float

    confidence_note: str


class Recommendation(TypedDict, total=False):
    """One business recommendation."""

    priority: Literal[
        "high",
        "medium",
        "low",
    ]

    action: str

    rationale: str


class ReportResult(TypedDict, total=False):
    """Generated business-intelligence report."""

    title: str

    executive_summary: str

    findings: list[str]

    recommendations: list[Recommendation]

    sources: list[str]

    markdown: str


# ============================================================================
# AGENT STATE
# ============================================================================

class AgentState(TypedDict, total=False):
    """
    Shared LangGraph state.
    """

    messages: Annotated[
        list[BaseMessage],
        add_messages,
    ]

    user_query: str

    query_context: QueryContext

    # Tool-routing plan
    tool_plan: dict[str, bool]

    forecast: list[ForecastRecord]

    shap_explanation: list[
        ShapExplanation
    ]

    historical_data: HistoricalDataResult

    business_context: BusinessContextResult

    risk_assessment: RiskAssessment

    synthesis: str

    recommendations: list[
        Recommendation
    ]

    report: ReportResult

    sources: list[str]

    errors: list[str]