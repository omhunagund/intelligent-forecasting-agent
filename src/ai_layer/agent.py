"""
LangGraph AI Decision Agent
===========================

Initial Phase-4 agent architecture.

This first version validates the graph orchestration using the
already-tested deterministic tools.

Graph:

    START
      ↓
    orchestrator
      ↓
    collect_evidence
      ↓
    synthesize
      ↓
    generate_report
      ↓
     END

The synthesis node is intentionally deterministic for this first
integration test. The LLM reasoning layer will be added only after
the graph/state/tool integration is validated.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from src.ai_layer.state import AgentState
from src.ai_layer.tools import (
    get_latest_forecast,
    get_shap_explanation,
    retrieve_business_context,
    query_historical_data,
    assess_forecast_risk,
)
from src.ai_layer.report_tools import build_report

from src.ai_layer.query_router import (
    route_query_node,
)


# ============================================================================
# ORCHESTRATOR
# ============================================================================

def orchestrator(
    state: AgentState,
) -> dict:
    """
    Decide which deterministic tools are required for the
    current user request.

    Tool selection is based on the request_type produced by
    the natural-language query router.
    """

    query_context = state.get(
        "query_context",
        {},
    )

    if not query_context:
        return {
            "errors": [
                "query_context is missing."
            ]
        }

    required = [
        "request_type",
        "series_type",
        "series_id",
    ]

    missing = [
        field
        for field in required
        if not query_context.get(field)
    ]

    if missing:
        return {
            "errors": [
                "Missing query-context fields: "
                + ", ".join(missing)
            ]
        }

    request_type = query_context[
        "request_type"
    ]

    # ---------------------------------------------------------------
    # Default plan
    # ---------------------------------------------------------------

    tool_plan = {
        "forecast": False,
        "shap": False,
        "historical": False,
        "business_context": False,
        "risk": False,
    }

    # ---------------------------------------------------------------
    # Forecast request
    # ---------------------------------------------------------------

    if request_type == "forecast":

        tool_plan = {
            "forecast": True,
            "shap": False,
            "historical": False,
            "business_context": True,
            "risk": False,
        }

    # ---------------------------------------------------------------
    # Explanation request
    # ---------------------------------------------------------------

    elif request_type == "explanation":

        tool_plan = {
            "forecast": True,
            "shap": True,
            "historical": False,
            "business_context": True,
            "risk": False,
        }

    # ---------------------------------------------------------------
    # Comparison request
    # ---------------------------------------------------------------

    elif request_type == "comparison":

        tool_plan = {
            "forecast": True,
            "shap": False,
            "historical": True,
            "business_context": True,
            "risk": False,
        }

    # ---------------------------------------------------------------
    # Risk request
    # ---------------------------------------------------------------

    elif request_type == "risk":

        tool_plan = {
            "forecast": True,
            "shap": True,
            "historical": True,
            "business_context": True,
            "risk": True,
        }

    # ---------------------------------------------------------------
    # Report request
    # ---------------------------------------------------------------

    elif request_type == "report":

        tool_plan = {
            "forecast": True,
            "shap": True,
            "historical": True,
            "business_context": True,
            "risk": True,
        }

    # ---------------------------------------------------------------
    # General business question
    # ---------------------------------------------------------------

    else:

        tool_plan = {
            "forecast": False,
            "shap": False,
            "historical": False,
            "business_context": True,
            "risk": False,
        }

    return {
        "tool_plan": tool_plan,
        "sources": [],
        "errors": [],
    }


# ============================================================================
# EVIDENCE COLLECTION
# ============================================================================

def collect_evidence(
    state: AgentState,
) -> dict:
    """
    Execute only the tools required by the orchestrator's
    tool plan.
    """

    query_context = state[
        "query_context"
    ]

    tool_plan = state.get(
        "tool_plan",
        {},
    )

    series_type = query_context[
        "series_type"
    ]

    series_id = query_context[
        "series_id"
    ]

    horizon = query_context.get(
        "forecast_horizon",
        4,
    )

    comparison_period = query_context.get(
        "comparison_period",
        "12 weeks",
    )

    errors = list(
        state.get(
            "errors",
            [],
        )
    )

    sources: list[str] = []

    forecast = None
    shap_explanation = None
    historical_data = None
    business_context = None
    risk_assessment = None

    # ==================================================================
    # 1. FORECAST
    # ==================================================================

    if tool_plan.get(
        "forecast",
        False,
    ):

        try:

            forecast = (
                get_latest_forecast.invoke(
                    {
                        "series_type":
                            series_type,
                        "series_id":
                            series_id,
                        "horizon":
                            horizon,
                    }
                )
            )

        except Exception as exc:

            errors.append(
                f"Forecast retrieval failed: {exc}"
            )

    # ==================================================================
    # 2. SHAP
    # ==================================================================

    if tool_plan.get(
        "shap",
        False,
    ):

        if (
            forecast
            and forecast.get(
                "forecasts"
            )
        ):

            try:

                latest_forecast = (
                    forecast[
                        "forecasts"
                    ][-1]
                )

                shap_explanation = (
                    get_shap_explanation.invoke(
                        {
                            "series_type":
                                series_type,
                            "series_id":
                                series_id,
                            "forecast_timestamp":
                                latest_forecast[
                                    "timestamp"
                                ],
                        }
                    )
                )

            except Exception as exc:

                errors.append(
                    f"SHAP retrieval failed: {exc}"
                )

        else:

            errors.append(
                "SHAP retrieval skipped because "
                "forecast evidence was unavailable."
            )

    # ==================================================================
    # 3. HISTORICAL
    # ==================================================================

    if tool_plan.get(
        "historical",
        False,
    ):

        try:

            historical_data = (
                query_historical_data.invoke(
                    {
                        "series_type":
                            series_type,
                        "series_id":
                            series_id,
                        "comparison_period":
                            comparison_period,
                    }
                )
            )

        except Exception as exc:

            errors.append(
                f"Historical retrieval failed: {exc}"
            )

    # ==================================================================
    # 4. BUSINESS CONTEXT / RAG
    # ==================================================================

    if tool_plan.get(
        "business_context",
        False,
    ):

        try:

            business_context = (
                retrieve_business_context.invoke(
                    {
                        "query":
                            state[
                                "user_query"
                            ],
                        "top_k":
                            5,
                    }
                )
            )

        except Exception as exc:

            errors.append(
                f"Business-context retrieval failed: {exc}"
            )

    # ==================================================================
    # 5. RISK
    # ==================================================================

    if tool_plan.get(
        "risk",
        False,
    ):

        if (
            forecast
            and forecast.get(
                "forecasts"
            )
        ):

            try:

                latest = (
                    forecast[
                        "forecasts"
                    ][-1]
                )

                risk_assessment = (
                    assess_forecast_risk.invoke(
                        {
                            "series_type":
                                series_type,
                            "series_id":
                                series_id,
                            "forecast_revenue":
                                latest[
                                    "forecast_revenue"
                                ],
                            "lower_80":
                                latest[
                                    "lower_80"
                                ],
                            "upper_80":
                                latest[
                                    "upper_80"
                                ],
                        }
                    )
                )

            except Exception as exc:

                errors.append(
                    f"Risk assessment failed: {exc}"
                )

        else:

            errors.append(
                "Risk assessment skipped because "
                "forecast evidence was unavailable."
            )

    # ==================================================================
    # COLLECT SOURCES
    # ==================================================================

    if forecast:

        source = forecast.get(
            "source"
        )

        if source:
            sources.append(
                source
            )

    if shap_explanation:

        source = shap_explanation.get(
            "source"
        )

        if source:
            sources.append(
                source
            )

    if historical_data:

        source = historical_data.get(
            "source"
        )

        if source:
            sources.append(
                source
            )

    if business_context:

        sources.extend(
            business_context.get(
                "sources",
                [],
            )
        )

    return {
        "forecast":
            forecast,
        "shap_explanation":
            (
                [shap_explanation]
                if shap_explanation
                else []
            ),
        "historical_data":
            historical_data,
        "business_context":
            business_context,
        "risk_assessment":
            risk_assessment,
        "sources":
            list(
                dict.fromkeys(
                    sources
                )
            ),
        "errors":
            errors,
    }


# ============================================================================
# DETERMINISTIC SYNTHESIS
# ============================================================================

from src.ai_layer.reasoning import (
    reason,
)

def synthesize(
    state: AgentState,
) -> dict:
    """
    LLM-backed evidence synthesis.

    The LLM receives only verified project evidence already collected
    by the deterministic tools.
    """

    return reason(
        state
    )


# ============================================================================
# REPORT NODE
# ============================================================================

def generate_report_node(
    state: AgentState,
) -> dict:
    """Generate the final structured report."""

    query_context = state.get(
        "query_context",
        {},
    )

    series_type = query_context.get(
        "series_type",
        "unknown",
    )

    series_id = query_context.get(
        "series_id",
        "unknown",
    )

    title = (
        f"{series_type.title()} "
        f"{series_id} Forecast Report"
    )

    synthesis = state.get(
        "synthesis",
        "",
    )

    risk = state.get(
        "risk_assessment"
    )

    findings: list[str] = []

    if synthesis:
        findings.extend(
            synthesis.splitlines()
        )

    recommendations: list[dict] = []

    if risk:

        if risk["status"] == "alert":

            recommendations.append(
                {
                    "priority":
                        "high",
                    "action":
                        (
                            "Review forecast reliability "
                            "before using the forecast "
                            "for high-impact planning."
                        ),
                    "rationale":
                        (
                            "Project monitoring indicates "
                            "alert-level risk."
                        ),
                }
            )

        elif risk["status"] == "warning":

            recommendations.append(
                {
                    "priority":
                        "medium",
                    "action":
                        (
                            "Review the supporting forecast "
                            "evidence before making high-impact "
                            "decisions."
                        ),
                    "rationale":
                        (
                            "Project monitoring indicates "
                            "warning-level risk."
                        ),
                }
            )

    report = build_report(
        title=title,
        executive_summary=(
            synthesis
            if synthesis
            else (
                "No synthesis was generated."
            )
        ),
        forecast=state.get(
            "forecast"
        ),
        shap_explanation=(
            state[
                "shap_explanation"
            ][0]
            if state.get(
                "shap_explanation"
            )
            else None
        ),
        historical_data=state.get(
            "historical_data"
        ),
        business_context=state.get(
            "business_context"
        ),
        risk_assessment=state.get(
            "risk_assessment"
        ),
        findings=findings,
        recommendations=recommendations,
        sources=state.get(
            "sources",
            [],
        ),
    )

    return {
        "report": report
    }


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def build_agent_graph():
    """Build the LangGraph AI Decision Agent."""

    graph = StateGraph(
        AgentState
    )

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    graph.add_node(
        "route_query",
        route_query_node,
    )

    graph.add_node(
        "orchestrator",
        orchestrator,
    )

    graph.add_node(
        "collect_evidence",
        collect_evidence,
    )

    graph.add_node(
        "synthesize",
        synthesize,
    )

    graph.add_node(
        "generate_report",
        generate_report_node,
    )

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    graph.add_edge(
        START,
        "route_query",
    )

    graph.add_edge(
        "route_query",
        "orchestrator",
    )

    graph.add_edge(
        "orchestrator",
        "collect_evidence",
    )

    graph.add_edge(
        "collect_evidence",
        "synthesize",
    )

    graph.add_edge(
        "synthesize",
        "generate_report",
    )

    graph.add_edge(
        "generate_report",
        END,
    )

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------

    return graph.compile()


# ============================================================================
# SMOKE TEST
# ============================================================================

def run_agent(
    user_query: str,
) -> dict:
    """
    Production entrypoint for the Intelligent Forecasting Agent.

    Parameters
    ----------
    user_query:
        Natural-language business question.

    Returns
    -------
    dict
        Final LangGraph state containing the routing decision,
        collected evidence, reasoning, risk assessment, and report.
    """

    user_query = user_query.strip()

    if not user_query:
        raise ValueError(
            "user_query must not be empty."
        )

    graph = build_agent_graph()

    initial_state: AgentState = {
        "user_query":
            user_query,
        "messages": [],
        "sources": [],
        "errors": [],
    }

    result = graph.invoke(
        initial_state
    )

    errors = result.get(
        "errors",
        [],
    )

    if errors:
        raise RuntimeError(
            "Agent execution completed with errors:\n"
            + "\n".join(
                f"- {error}"
                for error in errors
            )
        )

    return result

def run_agent_smoke_test() -> None:
    """
    Validate the reusable run_agent() entrypoint with several
    natural-language queries.
    """

    print(
        "=== PRODUCTION AGENT ENTRYPOINT TEST ==="
    )

    test_queries = [
        "Why is North revenue forecast risky?",
        "What is the next 4 week revenue forecast for Automotive?",
        "Explain what is driving the overall revenue forecast.",
        "How has Beauty & Health forecast performance been recently?",
        "Give me a report on the South region.",
    ]

    for index, query in enumerate(
        test_queries,
        start=1,
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"TEST QUERY {index}"
        )

        print(
            f"Query: {query}"
        )

        print(
            "=" * 70
        )

        result = run_agent(
            query
        )

        print(
            "\nQuery context:"
        )

        print(
            result.get(
                "query_context",
                "N/A",
            )
        )

        print(
            "\nTool plan:"
        )

        print(
            result.get(
                "tool_plan",
                "N/A",
            )
        )

        print(
            "\nSynthesis:"
        )

        print(
            result.get(
                "synthesis",
                "N/A",
            )
        )

        print(
            "\nReport:"
        )

        report = result.get(
            "report"
        )

        if report:

            print(
                report.get(
                    "path",
                    "N/A",
                )
            )

        print(
            "\nErrors:"
        )

        errors = result.get(
            "errors",
            [],
        )

        if errors:

            for error in errors:
                print(
                    f"- {error}"
                )

        else:

            print(
                "None"
            )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "=== PRODUCTION AGENT ENTRYPOINT TEST PASSED ==="
    )

if __name__ == "__main__":
    run_agent_smoke_test()