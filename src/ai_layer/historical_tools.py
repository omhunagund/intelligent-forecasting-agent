"""
Historical Data Access Tools
============================

Deterministic historical-data access for the AI layer.

Sources
-------
Overall:
    reports/stage1_weighted_ensemble_predictions.csv

Category + Region:
    reports/secondary/secondary_predictions.parquet

This module performs exact structured numerical lookups.
It does not use an LLM or semantic retrieval.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

OVERALL_HISTORY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "stage1_weighted_ensemble_predictions.csv"
)

SECONDARY_HISTORY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "secondary"
    / "secondary_predictions.parquet"
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
# LOADERS
# ============================================================================

def load_overall_history() -> pd.DataFrame:
    """Load validated overall walk-forward predictions."""

    if not OVERALL_HISTORY_PATH.is_file():
        raise FileNotFoundError(
            "Overall historical forecast artifact not found:\n"
            f"{OVERALL_HISTORY_PATH}"
        )

    df = pd.read_csv(
        OVERALL_HISTORY_PATH
    )

    required_columns = {
        "timestamp",
        "actual",
        "predicted",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Overall historical artifact is missing "
            f"columns: {sorted(missing)}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df["actual"] = pd.to_numeric(
        df["actual"],
        errors="coerce",
    )

    df["predicted"] = pd.to_numeric(
        df["predicted"],
        errors="coerce",
    )

    df["absolute_error"] = (
        df["actual"]
        - df["predicted"]
    ).abs()

    return (
        df.sort_values(
            "timestamp"
        )
        .reset_index(drop=True)
    )


def load_secondary_history() -> pd.DataFrame:
    """Load validated category/region walk-forward predictions."""

    if not SECONDARY_HISTORY_PATH.is_file():
        raise FileNotFoundError(
            "Secondary historical forecast artifact not found:\n"
            f"{SECONDARY_HISTORY_PATH}"
        )

    df = pd.read_parquet(
        SECONDARY_HISTORY_PATH
    )

    required_columns = {
        "series_type",
        "series_id",
        "timestamp",
        "actual",
        "predicted",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Secondary historical artifact is missing "
            f"columns: {sorted(missing)}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df["actual"] = pd.to_numeric(
        df["actual"],
        errors="coerce",
    )

    df["predicted"] = pd.to_numeric(
        df["predicted"],
        errors="coerce",
    )

    df["absolute_error"] = (
        df["actual"]
        - df["predicted"]
    ).abs()

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
# PERIOD PARSING
# ============================================================================

def parse_comparison_period(
    comparison_period: str,
) -> int:
    """
    Convert a simple comparison-period string into a number of weeks.

    Supported examples:
        "4 weeks"
        "8 weeks"
        "12 weeks"
        "recent 4 weeks"
        "recent 12 weeks"
    """

    value = comparison_period.strip().lower()

    if not value:
        raise ValueError(
            "comparison_period must not be empty."
        )

    aliases = {
        "recent week": 1,
        "last week": 1,
        "recent 4 weeks": 4,
        "last 4 weeks": 4,
        "recent 8 weeks": 8,
        "last 8 weeks": 8,
        "recent 12 weeks": 12,
        "last 12 weeks": 12,
        "recent 26 weeks": 26,
        "last 26 weeks": 26,
    }

    if value in aliases:
        return aliases[value]

    parts = value.split()

    for index, part in enumerate(parts):

        if part.isdigit():

            number = int(
                part
            )

            if (
                index + 1
                < len(parts)
                and parts[index + 1].startswith(
                    "week"
                )
            ):
                if number < 1:
                    raise ValueError(
                        "Comparison period must be "
                        "at least one week."
                    )

                return number

    raise ValueError(
        "Unsupported comparison_period. "
        "Use values such as "
        "'4 weeks', '12 weeks', or "
        "'recent 12 weeks'."
    )


# ============================================================================
# CALCULATE HISTORICAL METRICS
# ============================================================================

def calculate_historical_metrics(
    df: pd.DataFrame,
) -> dict:
    """Calculate exact metrics for a historical subset."""

    if df.empty:
        return {
            "mae": None,
            "rmse": None,
            "mape": None,
        }

    actual = df[
        "actual"
    ].to_numpy(
        dtype=float
    )

    predicted = df[
        "predicted"
    ].to_numpy(
        dtype=float
    )

    errors = (
        actual
        - predicted
    )

    mae = float(
        abs(errors).mean()
    )

    rmse = float(
        (
            errors ** 2
        ).mean()
        ** 0.5
    )

    mask = (
        abs(actual)
        > 1e-8
    )

    if mask.any():

        mape = float(
            (
                abs(
                    errors[mask]
                    / actual[mask]
                ).mean()
            )
            * 100.0
        )

    else:
        mape = None

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }


# ============================================================================
# OVERALL HISTORICAL QUERY
# ============================================================================

def query_overall_history(
    weeks: int,
) -> dict:
    """Return the most recent validated overall historical window."""

    df = load_overall_history()

    if weeks > len(df):
        raise ValueError(
            f"Requested {weeks} weeks, "
            f"but only {len(df)} historical "
            "overall records are available."
        )

    selected = (
        df.tail(weeks)
        .copy()
        .reset_index(drop=True)
    )

    metrics = calculate_historical_metrics(
        selected
    )

    records = []

    for _, row in selected.iterrows():

        records.append(
            {
                "timestamp":
                    row[
                        "timestamp"
                    ].strftime(
                        "%Y-%m-%d"
                    ),
                "actual_revenue":
                    float(
                        row[
                            "actual"
                        ]
                    ),
                "forecast_revenue":
                    float(
                        row[
                            "predicted"
                        ]
                    ),
                "absolute_error":
                    float(
                        row[
                            "absolute_error"
                        ]
                    ),
            }
        )

    return {
        "series_type":
            "overall",
        "series_id":
            "overall",
        "comparison_period":
            f"{weeks} weeks",
        "period_start":
            selected[
                "timestamp"
            ].min().strftime(
                "%Y-%m-%d"
            ),
        "period_end":
            selected[
                "timestamp"
            ].max().strftime(
                "%Y-%m-%d"
            ),
        "records":
            records,
        "metrics":
            metrics,
        "source":
            str(
                OVERALL_HISTORY_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
    }


# ============================================================================
# SECONDARY HISTORICAL QUERY
# ============================================================================

def query_secondary_history(
    series_type: str,
    series_id: str,
    weeks: int,
) -> dict:
    """Return the most recent validated history for one secondary series."""

    df = load_secondary_history()

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
            f"No historical records found for "
            f"{series_type}/{series_id}.\n"
            f"Available {series_type} series: "
            f"{available}"
        )

    if weeks > len(matching):
        raise ValueError(
            f"Requested {weeks} weeks for "
            f"{series_type}/{series_id}, "
            f"but only {len(matching)} "
            "records are available."
        )

    selected = (
        matching
        .sort_values(
            "timestamp"
        )
        .tail(weeks)
        .reset_index(drop=True)
    )

    metrics = calculate_historical_metrics(
        selected
    )

    records = []

    for _, row in selected.iterrows():

        records.append(
            {
                "timestamp":
                    row[
                        "timestamp"
                    ].strftime(
                        "%Y-%m-%d"
                    ),
                "actual_revenue":
                    float(
                        row[
                            "actual"
                        ]
                    ),
                "forecast_revenue":
                    float(
                        row[
                            "predicted"
                        ]
                    ),
                "absolute_error":
                    float(
                        row[
                            "absolute_error"
                        ]
                    ),
            }
        )

    return {
        "series_type":
            series_type,
        "series_id":
            series_id,
        "comparison_period":
            f"{weeks} weeks",
        "period_start":
            selected[
                "timestamp"
            ].min().strftime(
                "%Y-%m-%d"
            ),
        "period_end":
            selected[
                "timestamp"
            ].max().strftime(
                "%Y-%m-%d"
            ),
        "records":
            records,
        "metrics":
            metrics,
        "source":
            str(
                SECONDARY_HISTORY_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
    }


# ============================================================================
# PUBLIC FUNCTION
# ============================================================================

def query_historical_data_value(
    series_type: str,
    series_id: str,
    comparison_period: str,
) -> dict:
    """
    Public structured historical-data query.

    Example:
        series_type="region"
        series_id="North"
        comparison_period="12 weeks"
    """

    validate_series_request(
        series_type,
        series_id,
    )

    weeks = parse_comparison_period(
        comparison_period
    )

    if series_type == "overall":

        return query_overall_history(
            weeks=weeks
        )

    return query_secondary_history(
        series_type=series_type,
        series_id=series_id,
        weeks=weeks,
    )