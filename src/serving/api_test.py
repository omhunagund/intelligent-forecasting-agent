"""
FastAPI Serving Layer Tests
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.serving.api import app


client = TestClient(
    app
)


def test_root() -> None:

    response = client.get(
        "/"
    )

    assert response.status_code == 200

    body = response.json()

    assert (
        body["status"]
        == "running"
    )


def test_health() -> None:

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    body = response.json()

    assert (
        body["status"]
        == "ok"
    )

    assert (
        "service"
        in body
    )


def test_forecast_overall() -> None:

    response = client.post(
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

    assert response.status_code == 200

    body = response.json()

    assert (
        body["series_type"]
        == "overall"
    )

    assert (
        body["series_id"]
        == "overall"
    )

    assert len(
        body["forecasts"]
    ) == 4


def test_forecast_category() -> None:

    response = client.post(
        "/forecast",
        json={
            "series_type":
                "category",
            "series_id":
                "Automotive",
            "horizon":
                4,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert (
        body["series_type"]
        == "category"
    )

    assert (
        body["series_id"]
        == "Automotive"
    )


def test_explanation() -> None:

    response = client.get(
        "/explanation",
        params={
            "series_type":
                "region",
            "series_id":
                "North",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert (
        body["series_type"]
        == "region"
    )

    assert (
        body["series_id"]
        == "North"
    )

    assert (
        "drivers_up"
        in body
    )

    assert (
        "drivers_down"
        in body
    )


def test_agent_query() -> None:

    response = client.post(
        "/agent/query",
        json={
            "query":
                "Why is North revenue forecast risky?"
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert (
        body["query_context"][
            "series_type"
        ]
        == "region"
    )

    assert (
        body["query_context"][
            "series_id"
        ]
        == "North"
    )

    assert (
        body["synthesis"]
    )


def test_latest_report() -> None:

    response = client.get(
        "/reports/latest"
    )

    # A report should already exist because the AI layer
    # was validated before starting the serving layer.
    assert response.status_code == 200

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


def test_invalid_forecast_series() -> None:

    response = client.post(
        "/forecast",
        json={
            "series_type":
                "region",
            "series_id":
                "NotARegion",
            "horizon":
                4,
        },
    )

    assert response.status_code == 400


def main() -> None:

    print(
        "=== FASTAPI SERVING TESTS ==="
    )

    tests = [
        (
            "Root endpoint",
            test_root,
        ),
        (
            "Health endpoint",
            test_health,
        ),
        (
            "Overall forecast",
            test_forecast_overall,
        ),
        (
            "Category forecast",
            test_forecast_category,
        ),
        (
            "SHAP explanation",
            test_explanation,
        ),
        (
            "Agent query",
            test_agent_query,
        ),
        (
            "Latest report",
            test_latest_report,
        ),
        (
            "Invalid forecast",
            test_invalid_forecast_series,
        ),
    ]

    for name, test in tests:

        print(
            f"\n{name}..."
        )

        test()

        print(
            f"{name}: PASS"
        )

    print(
        "\n=== FASTAPI SERVING TESTS PASSED ==="
    )


if __name__ == "__main__":
    main()