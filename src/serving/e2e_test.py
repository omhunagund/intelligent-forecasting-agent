"""
Serving Layer End-to-End Integration Test
==========================================

Validates the deployed Dockerized serving stack through the public
FastAPI interface.

Expected architecture:

Stream/API boundary
    ↓
FastAPI
    ↓
ML + AI layers
    ↓
Forecast / SHAP / RAG / Risk / Groq
"""

from __future__ import annotations

import sys
from typing import Any

import requests


BASE_URL = "http://localhost:8000"


def get(
    endpoint: str,
    **kwargs: Any,
) -> requests.Response:

    response = requests.get(
        f"{BASE_URL}{endpoint}",
        timeout=120,
        **kwargs,
    )

    response.raise_for_status()

    return response


def post(
    endpoint: str,
    **kwargs: Any,
) -> requests.Response:

    response = requests.post(
        f"{BASE_URL}{endpoint}",
        timeout=180,
        **kwargs,
    )

    response.raise_for_status()

    return response


def test_health() -> None:

    response = get(
        "/health"
    )

    body = response.json()

    assert (
        body["status"]
        == "ok"
    )

    assert (
        body["monitoring_status"]
        == "alert"
    )

    print(
        "Health endpoint: PASS"
    )


def test_forecast() -> None:

    response = post(
        "/forecast",
        json={
            "series_type":
                "overall",
            "series_id":
                "overall",
            "horizon":
                4,
        },
    )

    body = response.json()

    assert (
        body["series_type"]
        == "overall"
    )

    assert (
        body["series_id"]
        == "overall"
    )

    assert (
        body["model"]
        == "xgboost"
    )

    assert len(
        body["forecasts"]
    ) == 4

    print(
        "Forecast endpoint: PASS"
    )


def test_explanation() -> None:

    response = get(
        "/explanation",
        params={
            "series_type":
                "overall",
            "series_id":
                "overall",
            "forecast_timestamp":
                "2018-09-23",
        },
    )

    body = response.json()

    assert (
        body["series_type"]
        == "overall"
    )

    assert (
        body["series_id"]
        == "overall"
    )

    assert (
        body["forecast_timestamp"]
        == "2018-09-23"
    )

    assert body[
        "drivers_up"
    ]

    assert body[
        "drivers_down"
    ]

    print(
        "SHAP endpoint: PASS"
    )


def test_agent_query() -> None:

    response = post(
        "/agent/query",
        json={
            "query":
                "Why is North revenue forecast risky?"
        },
    )

    body = response.json()

    context = body[
        "query_context"
    ]

    assert (
        context["request_type"]
        == "risk"
    )

    assert (
        context["series_type"]
        == "region"
    )

    assert (
        context["series_id"]
        == "North"
    )

    assert body[
        "synthesis"
    ]

    risk = body.get(
        "risk_assessment"
    )

    assert risk is not None

    assert (
        risk["status"]
        == "alert"
    )

    assert (
        risk["score"]
        >= 0
    )

    assert (
        risk["score"]
        <= 100
    )

    assert body[
        "sources"
    ]

    print(
        "Agent endpoint: PASS"
    )


def test_latest_report() -> None:

    response = get(
        "/reports/latest"
    )

    body = response.json()

    assert (
        body["filename"]
    )

    assert (
        body["content"]
    )

    assert (
        "Executive Summary"
        in body["content"]
    )

    assert (
        "Forecast Risk"
        in body["content"]
    )

    assert (
        "Sources"
        in body["content"]
    )

    print(
        "Latest report endpoint: PASS"
    )


def main() -> None:

    print(
        "=== DOCKERIZED SERVING E2E TEST ==="
    )

    tests = [
        (
            "Health",
            test_health,
        ),
        (
            "Forecast",
            test_forecast,
        ),
        (
            "SHAP",
            test_explanation,
        ),
        (
            "Agent",
            test_agent_query,
        ),
        (
            "Latest report",
            test_latest_report,
        ),
    ]

    for name, test in tests:

        print(
            f"\n{name}..."
        )

        test()

    print(
        "\n=== DOCKERIZED SERVING E2E TEST PASSED ==="
    )


if __name__ == "__main__":

    try:
        main()

    except Exception as exc:

        print(
            "\n=== DOCKERIZED SERVING E2E TEST FAILED ==="
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        sys.exit(1)