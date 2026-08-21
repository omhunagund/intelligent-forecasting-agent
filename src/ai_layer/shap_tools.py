"""
SHAP Data Access Tools
======================

Deterministic access to the project-generated TreeSHAP explanations.

Source:
    reports/shap/agent_shap_explanations.json

This module contains no LLM logic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

SHAP_EXPLANATIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "shap"
    / "agent_shap_explanations.json"
)


# ============================================================================
# VALID SERIES TYPES
# ============================================================================

VALID_SERIES_TYPES = {
    "overall",
    "category",
    "region",
}


# ============================================================================
# LOAD SHAP DATA
# ============================================================================

def load_shap_explanations() -> list[dict]:
    """Load the agent-ready SHAP explanation records."""

    if not SHAP_EXPLANATIONS_PATH.is_file():
        raise FileNotFoundError(
            "Agent-ready SHAP explanation file not found:\n"
            f"{SHAP_EXPLANATIONS_PATH}"
        )

    with open(
        SHAP_EXPLANATIONS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "SHAP explanation file must contain a list."
        )

    if not data:
        raise ValueError(
            "SHAP explanation file is empty."
        )

    return data


# ============================================================================
# VALIDATION
# ============================================================================

def validate_series_request(
    series_type: str,
    series_id: str,
) -> None:
    """Validate the requested series."""

    if series_type not in VALID_SERIES_TYPES:
        raise ValueError(
            "Invalid series_type. "
            f"Expected one of "
            f"{sorted(VALID_SERIES_TYPES)}, "
            f"received '{series_type}'."
        )

    if not series_id.strip():
        raise ValueError(
            "series_id must not be empty."
        )

    if (
        series_type == "overall"
        and series_id != "overall"
    ):
        raise ValueError(
            "For series_type='overall', "
            "series_id must be 'overall'."
        )


# ============================================================================
# NORMALIZE DRIVERS
# ============================================================================

def normalize_driver(
    driver: dict,
) -> dict:
    """Normalize one SHAP driver record."""

    feature = driver.get(
        "feature"
    )

    if not feature:
        raise ValueError(
            "SHAP driver is missing "
            "'feature'."
        )

    feature_value = driver.get(
        "feature_value"
    )

    shap_value = driver.get(
        "shap_value"
    )

    return {
        "feature":
            str(feature),
        "feature_value":
            None
            if feature_value is None
            else float(
                feature_value
            ),
        "shap_value":
            None
            if shap_value is None
            else float(
                shap_value
            ),
        "direction":
            (
                "up"
                if shap_value is not None
                and shap_value > 0
                else (
                    "down"
                    if shap_value is not None
                    and shap_value < 0
                    else "neutral"
                )
            ),
    }


# ============================================================================
# FIND EXPLANATIONS
# ============================================================================

def find_explanations(
    series_type: str,
    series_id: str,
    forecast_timestamp: str | None = None,
) -> list[dict]:
    """
    Find SHAP explanations for a series.

    If forecast_timestamp is provided, return the exact matching
    forecast date.

    Otherwise, return all available forecast explanations for
    the requested series.
    """

    validate_series_request(
        series_type,
        series_id,
    )

    data = load_shap_explanations()

    matches = [
        item
        for item in data
        if (
            item.get("series_type")
            == series_type
        )
        and (
            item.get("series_id")
            == series_id
        )
    ]

    if not matches:
        raise ValueError(
            f"No SHAP explanation found for "
            f"{series_type}/{series_id}."
        )

    if forecast_timestamp is not None:

        requested_date = pd.Timestamp(
            forecast_timestamp
        ).strftime(
            "%Y-%m-%d"
        )

        matches = [
            item
            for item in matches
            if pd.Timestamp(
                item[
                    "forecast_timestamp"
                ]
            ).strftime(
                "%Y-%m-%d"
            )
            == requested_date
        ]

        if not matches:
            available_dates = sorted(
                {
                    pd.Timestamp(
                        item[
                            "forecast_timestamp"
                        ]
                    ).strftime(
                        "%Y-%m-%d"
                    )
                    for item in data
                    if (
                        item.get(
                            "series_type"
                        )
                        == series_type
                    )
                    and (
                        item.get(
                            "series_id"
                        )
                        == series_id
                    )
                }
            )

            raise ValueError(
                f"No SHAP explanation found for "
                f"{series_type}/{series_id} "
                f"on {requested_date}.\n"
                f"Available dates: "
                f"{available_dates}"
            )

    return matches


# ============================================================================
# BUILD STRUCTURED RESPONSE
# ============================================================================

def build_shap_response(
    item: dict,
) -> dict:
    """Convert a raw SHAP record into the agent-facing structure."""

    drivers_up = [
        normalize_driver(
            driver
        )
        for driver in item.get(
            "drivers_up",
            []
        )
    ]

    drivers_down = [
        normalize_driver(
            driver
        )
        for driver in item.get(
            "drivers_down",
            []
        )
    ]

    return {
        "series_type":
            str(
                item[
                    "series_type"
                ]
            ),
        "series_id":
            str(
                item[
                    "series_id"
                ]
            ),
        "forecast_timestamp":
            pd.Timestamp(
                item[
                    "forecast_timestamp"
                ]
            ).strftime(
                "%Y-%m-%d"
            ),
        "forecast_revenue":
            float(
                item[
                    "forecast_revenue"
                ]
            ),
        "base_value":
            float(
                item[
                    "base_value"
                ]
            ),
        "drivers_up":
            drivers_up,
        "drivers_down":
            drivers_down,
        "source":
            str(
                SHAP_EXPLANATIONS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
    }


# ============================================================================
# PUBLIC DATA FUNCTION
# ============================================================================

def get_shap_explanation_data(
    series_type: str,
    series_id: str,
    forecast_timestamp: str | None = None,
) -> dict:
    """
    Retrieve project-generated TreeSHAP explanations.

    When no timestamp is supplied, the latest available explanation
    for the requested series is returned.
    """

    matches = find_explanations(
        series_type=series_type,
        series_id=series_id,
        forecast_timestamp=forecast_timestamp,
    )

    if forecast_timestamp is None:

        matches = sorted(
            matches,
            key=lambda item: pd.Timestamp(
                item[
                    "forecast_timestamp"
                ]
            ),
        )

        selected = matches[-1]

    else:
        selected = matches[0]

    return build_shap_response(
        selected
    )