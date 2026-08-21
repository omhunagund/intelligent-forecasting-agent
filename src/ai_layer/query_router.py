"""
Natural-Language Query Router
=============================

Converts a normal user question into the structured QueryContext
required by the LangGraph evidence-collection stage.

The router uses the same Groq reasoning model as the synthesis node,
but asks for a small structured JSON object instead of business prose.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.ai_layer.llm import get_reasoning_llm
from src.ai_layer.state import AgentState


ROUTER_SYSTEM_PROMPT = """
You are the query-routing component of an Intelligent Forecasting Agent.

Your task is ONLY to convert the user's natural-language request into
a structured routing object.

Do not answer the business question.
Do not calculate forecasts.
Do not invent facts.

Return ONLY valid JSON with these fields:

{
  "request_type": "...",
  "series_type": "...",
  "series_id": "...",
  "forecast_horizon": 4,
  "comparison_period": "12 weeks"
}

Allowed request_type values:
- forecast
- explanation
- comparison
- risk
- report
- general_business_question

Allowed series_type values:
- overall
- category
- region

Routing guidance:

1. "overall", "overall revenue", "total revenue", "company revenue"
   refers to:
   series_type = "overall"
   series_id = "overall"

2. If the user names one of the known project regions, use:
   series_type = "region"

3. If the user names one of the known project categories, use:
   series_type = "category"

4. Questions containing words such as:
   "risky", "risk", "reliable", "should I trust"
   generally map to request_type = "risk"

5. Questions containing:
   "why", "driver", "driving", "explain"
   generally map to request_type = "explanation"

6. Questions asking for predicted/future revenue generally map to:
   request_type = "forecast"

7. Questions comparing historical/current/model performance generally map to:
   request_type = "comparison"

8. Questions explicitly asking for a report/summary generally map to:
   request_type = "report"

9. If the request cannot be classified confidently, use:
   request_type = "general_business_question"

Never invent a category or region name.
When the scope is ambiguous, use the exact series name only when it
appears in the user's request.

Return JSON only.
"""


KNOWN_CATEGORIES = [
    "Automotive",
    "Beauty & Health",
    "Books & Media",
    "Electronics & Computing",
    "Fashion & Accessories",
    "Food & Beverage",
    "Gifts, Arts & Seasonal",
    "Home & Furniture",
    "Home Improvement & Garden",
    "Kids & Baby",
    "Kitchen & Appliances",
    "Office, Business & Services",
    "Pet Supplies",
    "Phones & Telecom",
    "Sports & Leisure",
]


KNOWN_REGIONS = [
    "Central-West",
    "North",
    "Northeast",
    "South",
    "Southeast",
]


def normalize_series_name(
    value: str,
) -> str:
    """Normalize whitespace while preserving project naming."""

    return re.sub(
        r"\s+",
        " ",
        value.strip(),
    )


def parse_router_json(
    content: str,
) -> dict[str, Any]:
    """Extract and validate the router's JSON object."""

    content = content.strip()

    # Handle accidental markdown fences.
    if content.startswith("```"):

        content = re.sub(
            r"^```(?:json)?\s*",
            "",
            content,
            flags=re.IGNORECASE,
        )

        content = re.sub(
            r"\s*```$",
            "",
            content,
        )

    try:
        result = json.loads(
            content
        )

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Query router returned invalid JSON:\n"
            f"{content}"
        ) from exc

    required_fields = {
        "request_type",
        "series_type",
        "series_id",
        "forecast_horizon",
        "comparison_period",
    }

    missing = (
        required_fields
        - set(result.keys())
    )

    if missing:
        raise ValueError(
            "Query router response is missing "
            f"fields: {sorted(missing)}"
        )

    allowed_request_types = {
        "forecast",
        "explanation",
        "comparison",
        "risk",
        "report",
        "general_business_question",
    }

    allowed_series_types = {
        "overall",
        "category",
        "region",
    }

    if result[
        "request_type"
    ] not in allowed_request_types:

        raise ValueError(
            "Invalid request_type returned by router: "
            f"{result['request_type']}"
        )

    if result[
        "series_type"
    ] not in allowed_series_types:

        raise ValueError(
            "Invalid series_type returned by router: "
            f"{result['series_type']}"
        )

    result["series_id"] = normalize_series_name(
        str(
            result["series_id"]
        )
    )

    if result["series_type"] == "overall":

        result["series_id"] = "overall"

    return result


def validate_known_series(
    routing: dict[str, Any],
) -> dict[str, Any]:
    """
    Prevent the LLM from inventing project series.

    If the router names a category/region that is not in our known
    project series, raise an explicit error.
    """

    series_type = routing[
        "series_type"
    ]

    series_id = routing[
        "series_id"
    ]

    if series_type == "category":

        if series_id not in KNOWN_CATEGORIES:

            raise ValueError(
                f"Unknown category '{series_id}'. "
                "The router must use one of the project's "
                f"known categories: {KNOWN_CATEGORIES}"
            )

    elif series_type == "region":

        if series_id not in KNOWN_REGIONS:

            raise ValueError(
                f"Unknown region '{series_id}'. "
                "The router must use one of the project's "
                f"known regions: {KNOWN_REGIONS}"
            )

    elif series_type == "overall":

        if series_id != "overall":

            raise ValueError(
                "Overall series must use series_id='overall'."
            )

    return routing


def route_user_query(
    query: str,
) -> dict[str, Any]:
    """Route one natural-language user query."""

    query = query.strip()

    if not query:
        raise ValueError(
            "User query must not be empty."
        )

    llm = get_reasoning_llm()

    response = llm.invoke(
        [
            SystemMessage(
                content=ROUTER_SYSTEM_PROMPT
            ),
            HumanMessage(
                content=query
            ),
        ]
    )

    content = (
        response.content
        if isinstance(
            response.content,
            str,
        )
        else str(
            response.content
        )
    )

    routing = parse_router_json(
        content
    )

    routing = validate_known_series(
        routing
    )

    return routing


def route_query_node(
    state: AgentState,
) -> dict:
    """
    LangGraph node that converts the user query into QueryContext.
    """

    query = state.get(
        "user_query",
        "",
    )

    routing = route_user_query(
        query
    )

    return {
        "query_context":
            routing
    }