"""
Production Monitoring Layer
===========================

Implements the Phase-3 monitoring requirements:

1. Data drift detection
   - Kolmogorov-Smirnov (KS) statistic
   - Population Stability Index (PSI)

2. Model performance monitoring
   - MAE
   - RMSE
   - MAPE
   - alert status based on configurable thresholds

The monitoring code is designed to be reusable when new production
data arrives. The standalone main() performs a historical smoke test
using the canonical feature dataset and the already-generated
walk-forward predictions.

Outputs
-------
reports/monitoring/
    data_drift_report.csv
    model_performance_report.csv
    monitoring_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

FEATURE_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "forecasting_features.parquet"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "monitoring"
)

ENSEMBLE_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "stage1_weighted_ensemble_predictions.csv"
)

SECONDARY_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "secondary"
    / "secondary_predictions.parquet"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Data drift
KS_P_VALUE_THRESHOLD = 0.05
PSI_WARNING_THRESHOLD = 0.10
PSI_ALERT_THRESHOLD = 0.25

# Model performance
MAPE_WARNING_THRESHOLD = 25.0
MAPE_ALERT_THRESHOLD = 35.0

RMSE_WARNING_MULTIPLIER = 1.25
RMSE_ALERT_MULTIPLIER = 1.50

MAE_WARNING_MULTIPLIER = 1.25
MAE_ALERT_MULTIPLIER = 1.50

# Historical reference/current windows used by the smoke test.
REFERENCE_WEEKS = 26
CURRENT_WEEKS = 26

# Features that are meaningful for numerical drift monitoring.
DRIFT_FEATURES = [
    "target_revenue",
    "lag_1",
    "lag_4",
    "rolling_mean_4",
    "rolling_std_4",
]


# ============================================================================
# DIRECTORY SETUP
# ============================================================================

def ensure_directories() -> None:
    """Create monitoring report directory."""

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================================
# DATA LOADING
# ============================================================================

def load_feature_dataset() -> pd.DataFrame:
    """Load the canonical feature dataset."""

    if not FEATURE_DATASET_PATH.is_file():
        raise FileNotFoundError(
            "Feature dataset not found:\n"
            f"{FEATURE_DATASET_PATH}"
        )

    df = pd.read_parquet(
        FEATURE_DATASET_PATH
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
# KS TEST
# ============================================================================

def calculate_ks_drift(
    reference: pd.Series,
    current: pd.Series,
) -> tuple[float, float]:
    """
    Calculate the two-sample Kolmogorov-Smirnov statistic
    and p-value.
    """

    reference_values = (
        pd.to_numeric(
            reference,
            errors="coerce",
        )
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    current_values = (
        pd.to_numeric(
            current,
            errors="coerce",
        )
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    if len(reference_values) == 0:
        raise ValueError(
            "Reference sample is empty."
        )

    if len(current_values) == 0:
        raise ValueError(
            "Current sample is empty."
        )

    statistic, p_value = ks_2samp(
        reference_values,
        current_values,
    )

    return (
        float(statistic),
        float(p_value),
    )


# ============================================================================
# PSI
# ============================================================================

def calculate_psi(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> float:
    """
    Calculate Population Stability Index (PSI).

    Bins are derived from reference quantiles so that the reference
    distribution remains the baseline.
    """

    reference_values = (
        pd.to_numeric(
            reference,
            errors="coerce",
        )
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    current_values = (
        pd.to_numeric(
            current,
            errors="coerce",
        )
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    if len(reference_values) == 0:
        raise ValueError(
            "Reference sample is empty."
        )

    if len(current_values) == 0:
        raise ValueError(
            "Current sample is empty."
        )

    # Remove duplicate quantile edges.
    quantile_edges = np.quantile(
        reference_values,
        np.linspace(
            0.0,
            1.0,
            bins + 1,
        ),
    )

    quantile_edges = np.unique(
        quantile_edges
    )

    if len(quantile_edges) < 2:
        return 0.0

    # Expand boundaries to capture equal-to-boundary values.
    quantile_edges[0] = -np.inf
    quantile_edges[-1] = np.inf

    reference_counts, _ = np.histogram(
        reference_values,
        bins=quantile_edges,
    )

    current_counts, _ = np.histogram(
        current_values,
        bins=quantile_edges,
    )

    reference_proportions = (
        reference_counts
        / max(
            reference_counts.sum(),
            1,
        )
    )

    current_proportions = (
        current_counts
        / max(
            current_counts.sum(),
            1,
        )
    )

    # Avoid log(0).
    epsilon = 1e-6

    reference_proportions = np.clip(
        reference_proportions,
        epsilon,
        None,
    )

    current_proportions = np.clip(
        current_proportions,
        epsilon,
        None,
    )

    psi = np.sum(
        (
            current_proportions
            - reference_proportions
        )
        * np.log(
            current_proportions
            / reference_proportions
        )
    )

    return float(psi)


# ============================================================================
# DRIFT CLASSIFICATION
# ============================================================================

def classify_drift(
    ks_p_value: float,
    psi: float,
) -> str:
    """
    Classify drift based on the PDF-aligned KS/PSI approach.

    alert:
        statistically significant KS drift and/or high PSI

    warning:
        moderate PSI or KS significance

    stable:
        neither metric indicates meaningful drift
    """

    if (
        ks_p_value < KS_P_VALUE_THRESHOLD
        or psi >= PSI_ALERT_THRESHOLD
    ):
        return "alert"

    if (
        psi >= PSI_WARNING_THRESHOLD
        or ks_p_value < 0.10
    ):
        return "warning"

    return "stable"


# ============================================================================
# ONE-SERIES DRIFT CHECK
# ============================================================================

def monitor_series_drift(
    series_df: pd.DataFrame,
    series_type: str,
    series_id: str,
    reference_weeks: int = 26,
    current_weeks: int = 26,
) -> pd.DataFrame:
    """
    Compare the latest 12 weeks against the same 12-week
    seasonal period from 52 weeks earlier.

    Calendar-derived features are intentionally excluded from
    DRIFT_FEATURES. Features with insufficient valid observations
    are skipped safely.
    """

    series_df = (
        series_df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------
    # Current window: latest 12 observations
    # ---------------------------------------------------------------

    current_df = series_df.iloc[
        -current_weeks:
    ].copy()

    # ---------------------------------------------------------------
    # Reference window: same 12 observations, 52 weeks earlier
    # ---------------------------------------------------------------

    current_start_position = (
        len(series_df)
        - current_weeks
    )

    reference_start_position = (
        current_start_position
        - 52
    )

    reference_end_position = (
        reference_start_position
        + reference_weeks
    )

    if reference_start_position < 0:
        raise ValueError(
            f"{series_type}/{series_id}: "
            "not enough historical observations "
            "for seasonal drift monitoring."
        )

    reference_df = series_df.iloc[
        reference_start_position:
        reference_end_position
    ].copy()

    reference_df = (
        reference_df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    current_df = (
        current_df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if len(reference_df) != reference_weeks:
        raise ValueError(
            f"{series_type}/{series_id}: "
            f"expected {reference_weeks} reference rows, "
            f"found {len(reference_df)}."
        )

    if len(current_df) != current_weeks:
        raise ValueError(
            f"{series_type}/{series_id}: "
            f"expected {current_weeks} current rows, "
            f"found {len(current_df)}."
        )

    # ---------------------------------------------------------------
    # IMPORTANT: initialize rows BEFORE the feature loop
    # ---------------------------------------------------------------

    rows: list[dict] = []

    # ---------------------------------------------------------------
    # Feature-level drift checks
    # ---------------------------------------------------------------

    for feature in DRIFT_FEATURES:

        if feature not in series_df.columns:
            continue

        reference = reference_df[
            feature
        ]

        current = current_df[
            feature
        ]

        reference_valid = pd.to_numeric(
            reference,
            errors="coerce",
        ).dropna()

        current_valid = pd.to_numeric(
            current,
            errors="coerce",
        ).dropna()

        # Skip features without enough valid observations.
        if (
            len(reference_valid) < 5
            or len(current_valid) < 5
        ):
            continue

        ks_statistic, ks_p_value = (
            calculate_ks_drift(
                reference_valid,
                current_valid,
            )
        )

        psi = calculate_psi(
            reference_valid,
            current_valid,
        )

        status = classify_drift(
            ks_p_value,
            psi,
        )

        rows.append(
            {
                "series_type":
                    series_type,
                "series_id":
                    series_id,
                "feature":
                    feature,
                "reference_start":
                    reference_df[
                        "timestamp"
                    ].min(),
                "reference_end":
                    reference_df[
                        "timestamp"
                    ].max(),
                "current_start":
                    current_df[
                        "timestamp"
                    ].min(),
                "current_end":
                    current_df[
                        "timestamp"
                    ].max(),
                "comparison_type":
                    "same_period_previous_year",
                "ks_statistic":
                    ks_statistic,
                "ks_p_value":
                    ks_p_value,
                "psi":
                    psi,
                "status":
                    status,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

def calculate_performance_metrics(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Calculate standard forecast-performance metrics."""

    actual_array = np.asarray(
        actual,
        dtype=float,
    )

    predicted_array = np.asarray(
        predicted,
        dtype=float,
    )

    if len(actual_array) != len(
        predicted_array
    ):
        raise ValueError(
            "Actual and predicted arrays "
            "must have equal length."
        )

    errors = (
        actual_array
        - predicted_array
    )

    mae = float(
        np.mean(
            np.abs(errors)
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                errors ** 2
            )
        )
    )

    non_zero_mask = (
        np.abs(actual_array) > 1e-8
    )

    if non_zero_mask.any():

        mape = float(
            np.mean(
                np.abs(
                    errors[
                        non_zero_mask
                    ]
                    / actual_array[
                        non_zero_mask
                    ]
                )
            )
            * 100.0
        )

    else:
        mape = 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }


# ============================================================================
# PERFORMANCE ALERTS
# ============================================================================

def classify_performance(
    metrics: dict[str, float],
    baseline_metrics: dict[str, float],
) -> str:
    """
    Compare recent model performance with its established baseline.

    Alert thresholds:
        MAPE >= 35%
        RMSE >= 1.50 × baseline
        MAE  >= 1.50 × baseline

    Warning thresholds:
        MAPE >= 25%
        RMSE >= 1.25 × baseline
        MAE  >= 1.25 × baseline
    """

    if (
        metrics["mape"]
        >= MAPE_ALERT_THRESHOLD
        or metrics["rmse"]
        >= (
            baseline_metrics["rmse"]
            * RMSE_ALERT_MULTIPLIER
        )
        or metrics["mae"]
        >= (
            baseline_metrics["mae"]
            * MAE_ALERT_MULTIPLIER
        )
    ):
        return "alert"

    if (
        metrics["mape"]
        >= MAPE_WARNING_THRESHOLD
        or metrics["rmse"]
        >= (
            baseline_metrics["rmse"]
            * RMSE_WARNING_MULTIPLIER
        )
        or metrics["mae"]
        >= (
            baseline_metrics["mae"]
            * MAE_WARNING_MULTIPLIER
        )
    ):
        return "warning"

    return "stable"


# ============================================================================
# PERFORMANCE MONITORING — GENERIC
# ============================================================================

def monitor_model_performance(
    predictions: pd.DataFrame,
    baseline_window: int = 20,
    recent_window: int = 12,
) -> dict:
    """
    Compare recent forecast errors against an earlier baseline.

    Expected columns:
        timestamp
        actual
        predicted
    """

    required_columns = {
        "timestamp",
        "actual",
        "predicted",
    }

    missing = (
        required_columns
        - set(predictions.columns)
    )

    if missing:
        raise ValueError(
            "Prediction dataframe is missing "
            f"columns: {sorted(missing)}"
        )

    predictions = (
        predictions.sort_values(
            "timestamp"
        )
        .reset_index(drop=True)
    )

    if len(predictions) < (
        baseline_window
        + recent_window
    ):
        raise ValueError(
            "Not enough predictions for "
            "performance monitoring."
        )

    baseline = predictions.iloc[
        -(
            baseline_window
            + recent_window
        ):
        -recent_window
    ]

    recent = predictions.iloc[
        -recent_window:
    ]

    baseline_metrics = (
        calculate_performance_metrics(
            actual=baseline["actual"],
            predicted=baseline["predicted"],
        )
    )

    recent_metrics = (
        calculate_performance_metrics(
            actual=recent["actual"],
            predicted=recent["predicted"],
        )
    )

    status = classify_performance(
        recent_metrics,
        baseline_metrics,
    )

    return {
        "baseline_start":
            baseline["timestamp"].min(),
        "baseline_end":
            baseline["timestamp"].max(),
        "recent_start":
            recent["timestamp"].min(),
        "recent_end":
            recent["timestamp"].max(),
        "baseline_mae":
            baseline_metrics["mae"],
        "baseline_rmse":
            baseline_metrics["rmse"],
        "baseline_mape":
            baseline_metrics["mape"],
        "recent_mae":
            recent_metrics["mae"],
        "recent_rmse":
            recent_metrics["rmse"],
        "recent_mape":
            recent_metrics["mape"],
        "status":
            status,
    }


# ============================================================================
# LOAD HISTORICAL PREDICTIONS
# ============================================================================

def load_secondary_predictions() -> pd.DataFrame:
    """Load secondary-series out-of-fold predictions."""

    if not SECONDARY_PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(
            "Secondary prediction artifact not found:\n"
            f"{SECONDARY_PREDICTIONS_PATH}"
        )

    df = pd.read_parquet(
        SECONDARY_PREDICTIONS_PATH
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    return df


def load_ensemble_predictions() -> pd.DataFrame:
    """
    Load the leakage-safe overall ensemble predictions.

    This file is used for the monitoring smoke test because it is
    the saved validated overall forecast stream.
    """

    if not ENSEMBLE_PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(
            "Overall ensemble predictions not found:\n"
            f"{ENSEMBLE_PREDICTIONS_PATH}"
        )

    df = pd.read_csv(
        ENSEMBLE_PREDICTIONS_PATH
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    return df


# ============================================================================
# MAIN MONITORING PIPELINE
# ============================================================================

def run_monitoring() -> None:
    """
    Run a historical smoke test of the monitoring framework.

    In production, the same reusable functions should be called
    whenever new data and actuals arrive.
    """

    print(
        "=== ML MONITORING PIPELINE ==="
    )

    ensure_directories()

    # ------------------------------------------------------------------
    # Load feature data
    # ------------------------------------------------------------------

    print(
        "\nLoading canonical feature dataset..."
    )

    features = load_feature_dataset()

    print(
        f"Loaded {len(features):,} rows."
    )

    # ------------------------------------------------------------------
    # Select production-level reference/current windows
    # ------------------------------------------------------------------

    overall = features[
        (
            features["series_type"]
            == "overall"
        )
        & (
            features["series_id"]
            == "overall"
        )
    ].copy()

    overall = (
        overall.sort_values(
            "timestamp"
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # Overall data drift
    # ------------------------------------------------------------------

    print(
        "\n=== OVERALL DATA DRIFT ==="
    )

    overall_drift = monitor_series_drift(
        series_df=overall,
        series_type="overall",
        series_id="overall",
    )

    print(
        overall_drift[
            [
                "feature",
                "ks_statistic",
                "ks_p_value",
                "psi",
                "status",
            ]
        ].to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # Category + region drift
    # ------------------------------------------------------------------

    all_drift_frames: list[pd.DataFrame] = [
        overall_drift
    ]

    secondary_series = (
        features[
            features["series_type"].isin(
                [
                    "category",
                    "region",
                ]
            )
        ][
            [
                "series_type",
                "series_id",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "series_type",
                "series_id",
            ]
        )
    )

    print(
        "\n=== SECONDARY DATA DRIFT ==="
    )

    for _, row in secondary_series.iterrows():

        series_type = str(
            row["series_type"]
        )

        series_id = str(
            row["series_id"]
        )

        series_df = features[
            (
                features["series_type"]
                == series_type
            )
            & (
                features["series_id"]
                == series_id
            )
        ].copy()

        drift = monitor_series_drift(
            series_df=series_df,
            series_type=series_type,
            series_id=series_id,
        )

        all_drift_frames.append(
            drift
        )

        alert_count = int(
            (
                drift["status"]
                == "alert"
            ).sum()
        )

        warning_count = int(
            (
                drift["status"]
                == "warning"
            ).sum()
        )

        print(
            f"{series_type}/"
            f"{series_id} | "
            f"alerts={alert_count} | "
            f"warnings={warning_count}"
        )

    drift_report = pd.concat(
        all_drift_frames,
        ignore_index=True,
    )

    # ------------------------------------------------------------------
    # Overall model performance
    # ------------------------------------------------------------------

    print(
        "\n=== OVERALL MODEL PERFORMANCE ==="
    )

    ensemble_predictions = (
        load_ensemble_predictions()
    )

    overall_performance = (
        monitor_model_performance(
            predictions=ensemble_predictions,
        )
    )

    for key, value in (
        overall_performance.items()
    ):
        print(
            f"{key}: {value}"
        )

    # ------------------------------------------------------------------
    # Secondary model performance
    # ------------------------------------------------------------------

    print(
        "\n=== SECONDARY MODEL PERFORMANCE ==="
    )

    secondary_predictions = (
        load_secondary_predictions()
    )

    performance_rows: list[dict] = [
        {
            "series_type":
                "overall",
            "series_id":
                "overall",
            **overall_performance,
        }
    ]

    for (
        series_type,
        series_id,
    ), group in secondary_predictions.groupby(
        [
            "series_type",
            "series_id",
        ],
        sort=True,
    ):

        performance = (
            monitor_model_performance(
                predictions=group,
            )
        )

        performance_rows.append(
            {
                "series_type":
                    series_type,
                "series_id":
                    series_id,
                **performance,
            }
        )

        print(
            f"{series_type}/"
            f"{series_id} | "
            f"status={performance['status']} | "
            f"recent MAPE="
            f"{performance['recent_mape']:.2f}%"
        )

    performance_report = pd.DataFrame(
        performance_rows
    )

    # ------------------------------------------------------------------
    # Overall monitoring status
    # ------------------------------------------------------------------

    drift_alerts = int(
        (
            drift_report["status"]
            == "alert"
        ).sum()
    )

    drift_warnings = int(
        (
            drift_report["status"]
            == "warning"
        ).sum()
    )

    performance_alerts = int(
        (
            performance_report["status"]
            == "alert"
        ).sum()
    )

    performance_warnings = int(
        (
            performance_report["status"]
            == "warning"
        ).sum()
    )

    if (
        drift_alerts > 0
        or performance_alerts > 0
    ):
        overall_status = "alert"

    elif (
        drift_warnings > 0
        or performance_warnings > 0
    ):
        overall_status = "warning"

    else:
        overall_status = "stable"

    # ------------------------------------------------------------------
    # Save reports
    # ------------------------------------------------------------------

    drift_report_path = (
        REPORT_DIR
        / "data_drift_report.csv"
    )

    performance_report_path = (
        REPORT_DIR
        / "model_performance_report.csv"
    )

    summary_path = (
        REPORT_DIR
        / "monitoring_summary.json"
    )

    drift_report.to_csv(
        drift_report_path,
        index=False,
    )

    performance_report.to_csv(
        performance_report_path,
        index=False,
    )

    summary = {
        "monitoring_status":
            overall_status,
        "drift_alerts":
            drift_alerts,
        "drift_warnings":
            drift_warnings,
        "performance_alerts":
            performance_alerts,
        "performance_warnings":
            performance_warnings,
        "series_monitored":
            int(
                len(
                    secondary_series
                )
            )
            + 1,
        "drift_features_monitored":
            int(
                len(
                    DRIFT_FEATURES
                )
            ),
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            default=str,
        )

    # ------------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------------

    print(
        "\n=== MONITORING COMPLETE ==="
    )

    print(
        f"Overall monitoring status: "
        f"{overall_status}"
    )

    print(
        f"Drift alerts: "
        f"{drift_alerts}"
    )

    print(
        f"Drift warnings: "
        f"{drift_warnings}"
    )

    print(
        f"Performance alerts: "
        f"{performance_alerts}"
    )

    print(
        f"Performance warnings: "
        f"{performance_warnings}"
    )

    print(
        "\nOutputs:"
    )

    print(
        f"{drift_report_path}"
    )

    print(
        f"{performance_report_path}"
    )

    print(
        f"{summary_path}"
    )


if __name__ == "__main__":
    run_monitoring()