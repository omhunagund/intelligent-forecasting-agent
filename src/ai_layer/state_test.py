"""
Smoke test for AI-layer state and tool contracts.
"""

from src.ai_layer.state import AgentState
from src.ai_layer.tools import AGENT_TOOLS


def main() -> None:
    print(
        "=== AI STATE / TOOL CONTRACT TEST ==="
    )

    state: AgentState = {
        "user_query":
            "Why is North revenue forecast risky?",
        "query_context": {
            "request_type":
                "risk",
            "series_type":
                "region",
            "series_id":
                "North",
            "forecast_horizon":
                4,
        },
        "sources": [],
        "errors": [],
    }

    print(
        "\nState initialized successfully."
    )

    print(
        f"User query: "
        f"{state['user_query']}"
    )

    print(
        "\nRegistered tools:"
    )

    for tool in AGENT_TOOLS:
        print(
            f"- {tool.name}"
        )

    if len(AGENT_TOOLS) != 6:
        raise AssertionError(
            "Expected exactly 6 agent tools."
        )

    expected_tools = {
        "get_latest_forecast",
        "get_shap_explanation",
        "retrieve_business_context",
        "query_historical_data",
        "assess_forecast_risk",
        "generate_report",
    }

    actual_tools = {
        tool.name
        for tool in AGENT_TOOLS
    }

    if actual_tools != expected_tools:
        raise AssertionError(
            "Tool registry mismatch.\n"
            f"Expected: {expected_tools}\n"
            f"Actual: {actual_tools}"
        )

    print(
        "\nAll six tool contracts validated."
    )

    print(
        "\n=== CONTRACT TEST PASSED ==="
    )


if __name__ == "__main__":
    main()