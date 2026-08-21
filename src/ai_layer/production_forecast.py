"""
Overall Production Forecast
===========================

Creates the final production forecast for overall revenue using
the selected production XGBoost model.

This is separate from the Stage-1 model-comparison artifacts.

Outputs:
    src/ml_layer/models/overall_production_xgboost.json
    reports/stage1_overall_production_xgboost_forecast.csv

The saved forecast is the source used by the AI-layer
get_latest_forecast() tool for the overall series.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ml_layer.training import (
    TARGET_COLUMN,
    fit_xgb,
    recursive_xgb_forecast,
)
from src.ml_layer.secondary_training import (
    calibrate_series_interval,
)


# ============================================================================
# PATHS
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

MODEL_DIR = (
    PROJECT_ROOT
    / "src"
    / "ml_layer"
    / "models"
)

MODEL_PATH = (
    MODEL_DIR
    / "overall_production_xgboost.json"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
)

FORECAST_PATH = (
    REPORT_DIR
    / "stage1_overall_production_xgboost_forecast.csv"
)

FORECAST_HORIZON = 4


# ============================================================================
# DATA LOADING
# ============================================================================

def load_overall_series() -> pd.DataFrame:
    """Load the canonical overall series."""

    if not FEATURE_DATASET_PATH.is_file():
        raise FileNotFoundError(
            "Canonical feature dataset not found:\n"
            f"{FEATURE_DATASET_PATH}"
        )

    features = pd.read_parquet(
        FEATURE_DATASET_PATH
    )

    features["timestamp"] = pd.to_datetime(
        features["timestamp"]
    )

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
        overall
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if len(overall) != 87:
        raise ValueError(
            "Expected 87 overall weekly observations, "
            f"found {len(overall)}."
        )

    return overall


# ============================================================================
# PRODUCTION FORECAST
# ============================================================================

def build_production_forecast() -> pd.DataFrame:
    """
    Train final overall XGBoost on the complete approved history
    and recursively forecast the next four weeks.
    """

    overall = load_overall_series()

    print(
        "\n=== OVERALL PRODUCTION XGBOOST ==="
    )

    print(
        f"Training rows: {len(overall)}"
    )

    print(
        f"Training start: "
        f"{overall['timestamp'].min()}"
    )

    print(
        f"Training end: "
        f"{overall['timestamp'].max()}"
    )

    # ------------------------------------------------------------------
    # Final model
    # ------------------------------------------------------------------

    model = fit_xgb(
        overall
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_model(
        MODEL_PATH
    )

    print(
        "\nSaved production model:"
    )

    print(
        MODEL_PATH
    )

    # ------------------------------------------------------------------
    # Future timestamps
    # ------------------------------------------------------------------

    last_timestamp = pd.Timestamp(
        overall["timestamp"].max()
    )

    future_timestamps = pd.date_range(
        start=(
            last_timestamp
            + pd.Timedelta(
                weeks=1
            )
        ),
        periods=FORECAST_HORIZON,
        freq="W-SUN",
    )

    # ------------------------------------------------------------------
    # True recursive forecast
    # ------------------------------------------------------------------

    predictions = (
        recursive_xgb_forecast(
            model=model,
            train_df=overall,
            future_timestamps=future_timestamps,
        )
    )

    # ------------------------------------------------------------------
    # Leakage-safe interval calibration
    # ------------------------------------------------------------------

    (
        calibration_actual,
        calibration_predicted,
    ) = calibrate_series_interval(
        train_df=overall
    )

    # Reproduce the project's residual interval logic.
    from src.ml_layer.training import (
        build_residual_interval,
    )

    lower_80, upper_80 = (
        build_residual_interval(
            calibration_actual=calibration_actual,
            calibration_predicted=calibration_predicted,
            future_predictions=predictions,
        )
    )

    # ------------------------------------------------------------------
    # Build result
    # ------------------------------------------------------------------

    forecast = pd.DataFrame(
        {
            "series_type":
                "overall",
            "series_id":
                "overall",
            "timestamp":
                future_timestamps,
            "forecast_revenue":
                predictions,
            "lower_80":
                lower_80,
            "upper_80":
                upper_80,
            "model":
                "xgboost",
            "model_artifact":
                str(
                    MODEL_PATH.relative_to(
                        PROJECT_ROOT
                    )
                ),
        }
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecast.to_csv(
        FORECAST_PATH,
        index=False,
    )

    return forecast


# ============================================================================
# MAIN
# ============================================================================

def run_production_forecast() -> None:
    """Generate and validate the production overall forecast."""

    forecast = build_production_forecast()

    print(
        "\n=== PRODUCTION FORECAST ==="
    )

    print(
        forecast.to_string(
            index=False
        )
    )

    if len(forecast) != FORECAST_HORIZON:
        raise RuntimeError(
            "Production forecast does not contain "
            f"{FORECAST_HORIZON} rows."
        )

    print(
        "\nForecast saved to:"
    )

    print(
        FORECAST_PATH
    )

    print(
        "\n=== PRODUCTION FORECAST COMPLETE ==="
    )


if __name__ == "__main__":
    run_production_forecast()