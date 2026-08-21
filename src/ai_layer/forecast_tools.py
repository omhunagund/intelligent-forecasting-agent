"""
Forecast Data Access Tools
==========================

Deterministic access to the saved production forecasting outputs.

Sources
-------
Overall:
    reports/stage1_weighted_ensemble_predictions.csv

Category + Region:
    reports/secondary/secondary_latest_forecasts.csv

This module contains no LLM logic.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

OVERALL_FORECAST_PATH = (
    PROJECT_ROOT
    / "reports"
    / "stage1_weighted_ensemble_predictions.csv"
)

SECONDARY_FORECAST_PATH = (
    PROJECT_ROOT
    / "reports"
    / "secondary"
    / "secondary_latest_forecasts.csv"
)


# ============================================================================
# VALID VALUES
# ============================================================================

VALID_SERIES_TYPES = {
    "overall",
    "category",
    "region",
}


# ============================================================================
# HELPERS
# ============================================================================

def validate_series_request(
    series_type: str,
    series_id: str,
) -> None:
    """Validate series type and ID."""

    if series_type not in VALID_SERIES_TYPES:
        raise ValueError(
            "Invalid series_type. "
            f"Expected one of {sorted(VALID_SERIES_TYPES)}, "
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


def load_overall_forecasts() -> pd.DataFrame:
    """Load the saved overall walk-forward ensemble predictions."""

    if not OVERALL_FORECAST_PATH.is_file():
        raise FileNotFoundError(
            "Overall forecast artifact not found:\n"
            f"{OVERALL_FORECAST_PATH}"
        )

    df = pd.read_csv(
        OVERALL_FORECAST_PATH
    )

    required_columns = {
        "timestamp",
        "actual",
        "predicted",
        "lower_80",
        "upper_80",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Overall forecast artifact is missing "
            f"columns: {sorted(missing)}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    return (
        df.sort_values(
            "timestamp"
        )
        .reset_index(drop=True)
    )


def load_secondary_forecasts() -> pd.DataFrame:
    """Load the saved category/region future forecasts."""

    if not SECONDARY_FORECAST_PATH.is_file():
        raise FileNotFoundError(
            "Secondary forecast artifact not found:\n"
            f"{SECONDARY_FORECAST_PATH}"
        )

    df = pd.read_csv(
        SECONDARY_FORECAST_PATH
    )

    required_columns = {
        "series_type",
        "series_id",
        "timestamp",
        "forecast_revenue",
        "lower_80",
        "upper_80",
        "model",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Secondary forecast artifact is missing "
            f"columns: {sorted(missing)}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    return (
        df.sort_values(
            [
                "series_type",
                "series_id",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================================
# OVERALL FORECAST
# ============================================================================

def get_overall_forecast(
    horizon: int = 4,
) -> dict:
    """
    Retrieve the latest production overall XGBoost forecast.
    """

    if horizon < 1:
        raise ValueError(
            "horizon must be at least 1."
        )

    production_path = (
        PROJECT_ROOT
        / "reports"
        / "stage1_overall_production_xgboost_forecast.csv"
    )

    if not production_path.is_file():
        raise FileNotFoundError(
            "Overall production XGBoost forecast "
            "has not been generated yet:\n"
            f"{production_path}"
        )

    df = pd.read_csv(
        production_path
    )

    required_columns = {
        "series_type",
        "series_id",
        "timestamp",
        "forecast_revenue",
        "lower_80",
        "upper_80",
        "model",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Production overall forecast is missing "
            f"columns: {sorted(missing)}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    latest = (
        df.sort_values(
            "timestamp"
        )
        .head(horizon)
        .reset_index(drop=True)
    )

    forecasts = []

    for _, row in latest.iterrows():

        forecasts.append(
            {
                "timestamp":
                    row[
                        "timestamp"
                    ].strftime(
                        "%Y-%m-%d"
                    ),
                "forecast_revenue":
                    float(
                        row[
                            "forecast_revenue"
                        ]
                    ),
                "lower_80":
                    float(
                        row[
                            "lower_80"
                        ]
                    ),
                "upper_80":
                    float(
                        row[
                            "upper_80"
                        ]
                    ),
                "model":
                    "xgboost",
            }
        )

    return {
        "series_type":
            "overall",
        "series_id":
            "overall",
        "model":
            "xgboost",
        "forecast_horizon":
            len(forecasts),
        "forecasts":
            forecasts,
        "source":
            str(
                production_path.relative_to(
                    PROJECT_ROOT
                )
            ),
    }


# ============================================================================
# CATEGORY / REGION FORECAST
# ============================================================================

def get_secondary_forecast(
    series_type: str,
    series_id: str,
    horizon: int = 4,
) -> dict:
    """Retrieve the latest forecast for one category or region."""

    if horizon < 1:
        raise ValueError(
            "horizon must be at least 1."
        )

    df = load_secondary_forecasts()

    matching = df[
        (
            df["series_type"]
            == series_type
        )
        & (
            df["series_id"]
            == series_id
        )
    ].copy()

    if matching.empty:
        available = (
            df.loc[
                df["series_type"]
                == series_type,
                "series_id",
            ]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        raise ValueError(
            f"No forecast found for "
            f"{series_type}/{series_id}.\n"
            f"Available {series_type} series: "
            f"{available}"
        )

    matching = (
        matching.sort_values(
            "timestamp"
        )
        .tail(horizon)
        .reset_index(drop=True)
    )

    forecasts = []

    for _, row in matching.iterrows():

        forecasts.append(
            {
                "timestamp":
                    row[
                        "timestamp"
                    ].strftime(
                        "%Y-%m-%d"
                    ),
                "forecast_revenue":
                    float(
                        row[
                            "forecast_revenue"
                        ]
                    ),
                "lower_80":
                    float(
                        row[
                            "lower_80"
                        ]
                    ),
                "upper_80":
                    float(
                        row[
                            "upper_80"
                        ]
                    ),
                "model":
                    str(
                        row[
                            "model"
                        ]
                    ),
            }
        )

    return {
        "series_type":
            series_type,
        "series_id":
            series_id,
        "model":
            str(
                matching[
                    "model"
                ].iloc[0]
            ),
        "forecast_horizon":
            len(forecasts),
        "forecasts":
            forecasts,
        "source":
            str(
                SECONDARY_FORECAST_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
    }


# ============================================================================
# PUBLIC TOOL FUNCTION
# ============================================================================

def get_latest_forecast_data(
    series_type: str,
    series_id: str,
    horizon: int = 4,
) -> dict:
    """
    Public deterministic forecast lookup.

    Returns a structured dictionary suitable for use by the
    LangChain tool wrapper.
    """

    validate_series_request(
        series_type,
        series_id,
    )

    if series_type == "overall":
        return get_overall_forecast(
            horizon=horizon
        )

    return get_secondary_forecast(
        series_type=series_type,
        series_id=series_id,
        horizon=horizon,
    )