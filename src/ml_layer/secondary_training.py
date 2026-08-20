"""
Secondary-series forecasting pipeline.

Models:
    - 15 business-category XGBoost models
    - 5 regional XGBoost models

Validation:
    - 52-week expanding training window
    - 4-week validation horizon
    - 4-week step
    - 8 walk-forward folds
    - true recursive forecasting

Outputs:
    - fold-level validation metrics
    - series-level summary metrics
    - out-of-fold predictions
    - latest 4-week forecasts
    - final XGBoost model artifacts
    - MLflow experiment tracking

The canonical forecasting features are loaded from:
    data/features/forecasting_features.parquet
"""

from __future__ import annotations

import re
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.data_layer.feature_engineering import (
    build_recursive_feature_row,
)
from src.ml_layer.evaluation import (
    aggregate_fold_metrics,
    calculate_metrics,
    generate_walk_forward_folds,
)
from src.ml_layer.training import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_residual_interval,
    fit_xgb,
    recursive_xgb_forecast,
)


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

MODEL_DIR = (
    PROJECT_ROOT
    / "src"
    / "ml_layer"
    / "models"
    / "secondary"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "secondary"
)

MLFLOW_DB_PATH = (
    PROJECT_ROOT
    / "mlflow.db"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

INITIAL_TRAIN_WEEKS = 52
VALIDATION_HORIZON = 4
STEP_WEEKS = 4

CALIBRATION_INITIAL_WEEKS = 20
CALIBRATION_HORIZON = 4
CALIBRATION_STEP = 4

EXPECTED_CATEGORY_COUNT = 15
EXPECTED_REGION_COUNT = 5


# ============================================================================
# DIRECTORY SETUP
# ============================================================================

def ensure_directories() -> None:
    """Create secondary ML output directories."""

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================================
# MLFLOW
# ============================================================================

def configure_mlflow() -> None:
    """Configure the same SQLite MLflow tracking backend."""

    tracking_uri = (
        f"sqlite:///"
        f"{MLFLOW_DB_PATH.as_posix()}"
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    mlflow.set_experiment(
        "intelligent-forecasting-agent"
    )

    print(
        f"MLflow tracking URI: "
        f"{tracking_uri}"
    )


# ============================================================================
# DATA LOADING
# ============================================================================

def load_feature_dataset() -> pd.DataFrame:
    """
    Load the canonical feature dataset.

    No feature regeneration happens here.
    """

    if not FEATURE_DATASET_PATH.is_file():
        raise FileNotFoundError(
            "Feature dataset not found:\n"
            f"{FEATURE_DATASET_PATH}\n\n"
            "Run the Data Layer pipeline first."
        )

    df = pd.read_parquet(
        FEATURE_DATASET_PATH
    )

    required_columns = {
        "timestamp",
        "series_type",
        "series_id",
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Feature dataset is missing "
            f"required columns: "
            f"{sorted(missing_columns)}"
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
# SECONDARY SERIES VALIDATION
# ============================================================================

def get_secondary_series(
    features: pd.DataFrame,
) -> dict[str, list[str]]:
    """
    Discover and validate the expected 15 category and 5 region series.
    """

    categories = sorted(
        features.loc[
            features["series_type"]
            == "category",
            "series_id",
        ]
        .dropna()
        .unique()
        .tolist()
    )

    regions = sorted(
        features.loc[
            features["series_type"]
            == "region",
            "series_id",
        ]
        .dropna()
        .unique()
        .tolist()
    )

    if len(categories) != EXPECTED_CATEGORY_COUNT:
        raise ValueError(
            "Expected "
            f"{EXPECTED_CATEGORY_COUNT} category series, "
            f"found {len(categories)}."
        )

    if len(regions) != EXPECTED_REGION_COUNT:
        raise ValueError(
            "Expected "
            f"{EXPECTED_REGION_COUNT} region series, "
            f"found {len(regions)}."
        )

    return {
        "category": categories,
        "region": regions,
    }


def get_single_series(
    features: pd.DataFrame,
    series_type: str,
    series_id: str,
) -> pd.DataFrame:
    """Return one complete time series."""

    series = features[
        (
            features["series_type"]
            == series_type
        )
        & (
            features["series_id"]
            == series_id
        )
    ].copy()

    series = (
        series.sort_values(
            "timestamp"
        )
        .reset_index(drop=True)
    )

    if len(series) != 87:
        raise ValueError(
            f"{series_type}/{series_id}: "
            f"expected 87 weekly rows, "
            f"found {len(series)}."
        )

    return series


# ============================================================================
# SERIES-SPECIFIC INTERVAL CALIBRATION
# ============================================================================

def calibrate_series_interval(
    train_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate multiple out-of-sample calibration residuals using
    nested rolling-origin XGBoost forecasts inside one outer
    training fold.

    This mirrors the leakage-safe interval calibration used by
    the flagship overall XGBoost model.
    """

    train_df = (
        train_df.sort_values(
            "timestamp"
        )
        .reset_index(drop=True)
    )

    calibration_actuals: list[float] = []
    calibration_predictions: list[float] = []

    train_size = len(train_df)

    origin = (
        CALIBRATION_INITIAL_WEEKS
    )

    while (
        origin
        + CALIBRATION_HORIZON
        <= train_size
    ):

        pseudo_train = train_df.iloc[
            :origin
        ].copy()

        pseudo_validation = train_df.iloc[
            origin:
            origin + CALIBRATION_HORIZON
        ].copy()

        model = fit_xgb(
            pseudo_train
        )

        future_timestamps = (
            pd.DatetimeIndex(
                pseudo_validation[
                    "timestamp"
                ]
            )
        )

        predictions = (
            recursive_xgb_forecast(
                model=model,
                train_df=pseudo_train,
                future_timestamps=future_timestamps,
            )
        )

        actual = (
            pseudo_validation[
                TARGET_COLUMN
            ]
            .to_numpy(
                dtype=float
            )
        )

        calibration_actuals.extend(
            actual.tolist()
        )

        calibration_predictions.extend(
            predictions.tolist()
        )

        origin += CALIBRATION_STEP

    if not calibration_actuals:
        raise ValueError(
            "Unable to generate interval "
            "calibration residuals."
        )

    return (
        np.asarray(
            calibration_actuals,
            dtype=float,
        ),
        np.asarray(
            calibration_predictions,
            dtype=float,
        ),
    )


# ============================================================================
# ONE SERIES — WALK-FORWARD VALIDATION
# ============================================================================

def evaluate_series(
    series_type: str,
    series_id: str,
    series_df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Run the complete 8-fold walk-forward XGBoost evaluation
    for one category or region series.
    """

    folds = (
        generate_walk_forward_folds(
            timestamps=series_df[
                "timestamp"
            ]
        )
    )

    fold_metric_rows: list[dict] = []
    prediction_rows: list[dict] = []

    for fold in folds:

        train_df = series_df[
            series_df["timestamp"]
            <= fold.train_end
        ].copy()

        validation_df = series_df[
            (
                series_df["timestamp"]
                >= fold.validation_start
            )
            & (
                series_df["timestamp"]
                <= fold.validation_end
            )
        ].copy()

        expected_train_size = (
            INITIAL_TRAIN_WEEKS
            + (
                fold.fold_number - 1
            )
            * STEP_WEEKS
        )

        if len(train_df) != expected_train_size:
            raise ValueError(
                f"{series_type}/{series_id}, "
                f"fold {fold.fold_number}: "
                f"expected "
                f"{expected_train_size} training rows, "
                f"found {len(train_df)}."
            )

        if len(validation_df) != VALIDATION_HORIZON:
            raise ValueError(
                f"{series_type}/{series_id}, "
                f"fold {fold.fold_number}: "
                f"expected "
                f"{VALIDATION_HORIZON} validation rows, "
                f"found {len(validation_df)}."
            )

        # ------------------------------------------------------------
        # Final model for this outer fold
        # ------------------------------------------------------------

        model = fit_xgb(
            train_df
        )

        # ------------------------------------------------------------
        # True recursive validation forecast
        # ------------------------------------------------------------

        future_timestamps = (
            pd.DatetimeIndex(
                validation_df[
                    "timestamp"
                ]
            )
        )

        predicted = (
            recursive_xgb_forecast(
                model=model,
                train_df=train_df,
                future_timestamps=future_timestamps,
            )
        )

        actual = (
            validation_df[
                TARGET_COLUMN
            ]
            .to_numpy(
                dtype=float
            )
        )

        # ------------------------------------------------------------
        # Leakage-safe interval calibration
        # ------------------------------------------------------------

        (
            calibration_actual,
            calibration_predicted,
        ) = calibrate_series_interval(
            train_df
        )

        lower, upper = (
            build_residual_interval(
                calibration_actual=calibration_actual,
                calibration_predicted=calibration_predicted,
                future_predictions=predicted,
            )
        )

        metrics = calculate_metrics(
            actual=actual,
            predicted=predicted,
            lower=lower,
            upper=upper,
        )

        fold_metric_rows.append(
            {
                "series_type":
                    series_type,
                "series_id":
                    series_id,
                "fold":
                    fold.fold_number,
                "train_start":
                    fold.train_start,
                "train_end":
                    fold.train_end,
                "validation_start":
                    fold.validation_start,
                "validation_end":
                    fold.validation_end,
                "mae":
                    metrics["mae"],
                "rmse":
                    metrics["rmse"],
                "mape":
                    metrics["mape"],
                "interval_coverage_pct":
                    metrics[
                        "interval_coverage_pct"
                    ],
            }
        )

        for index, timestamp in enumerate(
            validation_df[
                "timestamp"
            ]
        ):

            prediction_rows.append(
                {
                    "series_type":
                        series_type,
                    "series_id":
                        series_id,
                    "fold":
                        fold.fold_number,
                    "timestamp":
                        timestamp,
                    "actual":
                        float(
                            actual[index]
                        ),
                    "predicted":
                        float(
                            predicted[index]
                        ),
                    "lower_80":
                        float(
                            lower[index]
                        ),
                    "upper_80":
                        float(
                            upper[index]
                        ),
                }
            )

        print(
            f"    Fold "
            f"{fold.fold_number}/8 | "
            f"MAPE={metrics['mape']:.2f}% | "
            f"RMSE={metrics['rmse']:.2f} | "
            f"MAE={metrics['mae']:.2f}"
        )

    return (
        pd.DataFrame(
            fold_metric_rows
        ),
        pd.DataFrame(
            prediction_rows
        ),
    )


# ============================================================================
# FINAL MODEL + LATEST FORECAST
# ============================================================================

def slugify(value: str) -> str:
    """Create a safe filename component."""

    value = value.lower()

    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        value,
    )

    return value.strip("_")


def train_final_series_model(
    series_type: str,
    series_id: str,
    series_df: pd.DataFrame,
) -> dict:
    """
    Train the final XGBoost model on the complete 87-week series
    and generate the next four weekly forecasts.
    """

    model = fit_xgb(
        series_df
    )

    last_timestamp = pd.Timestamp(
        series_df[
            "timestamp"
        ].max()
    )

    future_timestamps = pd.date_range(
        start=(
            last_timestamp
            + pd.Timedelta(
                weeks=1
            )
        ),
        periods=4,
        freq="W-SUN",
    )

    predicted = (
        recursive_xgb_forecast(
            model=model,
            train_df=series_df,
            future_timestamps=future_timestamps,
        )
    )

    calibration_actual, calibration_predicted = (
        calibrate_series_interval(
            series_df
        )
    )

    lower, upper = (
        build_residual_interval(
            calibration_actual=calibration_actual,
            calibration_predicted=calibration_predicted,
            future_predictions=predicted,
        )
    )

    model_filename = (
        f"{slugify(series_type)}_"
        f"{slugify(series_id)}.json"
    )

    model_path = (
        MODEL_DIR
        / model_filename
    )

    model.save_model(
        model_path
    )

    forecast_rows = []

    for index, timestamp in enumerate(
        future_timestamps
    ):

        forecast_rows.append(
            {
                "series_type":
                    series_type,
                "series_id":
                    series_id,
                "timestamp":
                    timestamp,
                "forecast_revenue":
                    float(
                        predicted[index]
                    ),
                "lower_80":
                    float(
                        lower[index]
                    ),
                "upper_80":
                    float(
                        upper[index]
                    ),
                "model":
                    "xgboost",
                "model_artifact":
                    str(
                        model_path
                    ),
            }
        )

    return {
        "model": model,
        "model_path": model_path,
        "forecast":
            pd.DataFrame(
                forecast_rows
            ),
    }


# ============================================================================
# MLFLOW LOGGING
# ============================================================================

def log_series_mlflow(
    series_type: str,
    series_id: str,
    aggregate_metrics: dict[str, float],
    fold_metrics: pd.DataFrame,
    model_path: Path,
) -> None:
    """Log one secondary series model to MLflow."""

    safe_name = (
        f"{series_type}_"
        f"{slugify(series_id)}"
    )

    with mlflow.start_run(
        run_name=f"secondary_{safe_name}",
    ):

        mlflow.log_params(
            {
                "series_type":
                    series_type,
                "series_id":
                    series_id,
                "model":
                    "xgboost",
                "initial_train_weeks":
                    INITIAL_TRAIN_WEEKS,
                "validation_horizon":
                    VALIDATION_HORIZON,
                "step_weeks":
                    STEP_WEEKS,
                "recursive_forecasting":
                    True,
                "calibration_initial_weeks":
                    CALIBRATION_INITIAL_WEEKS,
                "calibration_horizon":
                    CALIBRATION_HORIZON,
                "calibration_step":
                    CALIBRATION_STEP,
            }
        )

        mlflow.log_metrics(
            {
                "mean_mae":
                    float(
                        aggregate_metrics[
                            "mean_mae"
                        ]
                    ),
                "mean_rmse":
                    float(
                        aggregate_metrics[
                            "mean_rmse"
                        ]
                    ),
                "mean_mape":
                    float(
                        aggregate_metrics[
                            "mean_mape"
                        ]
                    ),
                "mean_interval_coverage_pct":
                    float(
                        aggregate_metrics[
                            "mean_interval_coverage_pct"
                        ]
                    ),
            }
        )

        # Save a temporary fold report for MLflow.
        fold_report_path = (
            REPORT_DIR
            / (
                f"{safe_name}_"
                "fold_metrics.csv"
            )
        )

        fold_metrics.to_csv(
            fold_report_path,
            index=False,
        )

        mlflow.log_artifact(
            str(
                fold_report_path
            )
        )

        if model_path.is_file():
            mlflow.log_artifact(
                str(
                    model_path
                )
            )


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_secondary_training() -> None:
    """Train and evaluate all 20 secondary XGBoost series."""

    print(
        "=== SECONDARY SERIES ML PIPELINE ==="
    )

    ensure_directories()
    configure_mlflow()

    # ------------------------------------------------------------------
    # Load canonical features
    # ------------------------------------------------------------------

    print(
        "\nLoading canonical feature dataset..."
    )

    features = load_feature_dataset()

    print(
        f"Loaded {len(features):,} rows."
    )

    # ------------------------------------------------------------------
    # Discover expected series
    # ------------------------------------------------------------------

    series_groups = get_secondary_series(
        features
    )

    categories = (
        series_groups["category"]
    )

    regions = (
        series_groups["region"]
    )

    print(
        "\n=== SERIES DISCOVERY ==="
    )

    print(
        f"Category series: {len(categories)}"
    )

    print(
        f"Region series:   {len(regions)}"
    )

    print(
        f"Total secondary series: "
        f"{len(categories) + len(regions)}"
    )

    # ------------------------------------------------------------------
    # Containers for outputs
    # ------------------------------------------------------------------

    all_fold_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    latest_forecasts: list[pd.DataFrame] = []

    # ------------------------------------------------------------------
    # Categories
    # ------------------------------------------------------------------

    print(
        "\n=== CATEGORY XGBOOST MODELS ==="
    )

    for index, series_id in enumerate(
        categories,
        start=1,
    ):

        print(
            f"\n[{index}/{len(categories)}] "
            f"Category: {series_id}"
        )

        series_df = get_single_series(
            features=features,
            series_type="category",
            series_id=series_id,
        )

        (
            fold_metrics,
            predictions,
        ) = evaluate_series(
            series_type="category",
            series_id=series_id,
            series_df=series_df,
        )

        aggregate_metrics = (
            aggregate_fold_metrics(
                fold_metrics
            )
        )

        final_result = (
            train_final_series_model(
                series_type="category",
                series_id=series_id,
                series_df=series_df,
            )
        )

        log_series_mlflow(
            series_type="category",
            series_id=series_id,
            aggregate_metrics=aggregate_metrics,
            fold_metrics=fold_metrics,
            model_path=final_result[
                "model_path"
            ],
        )

        all_fold_metrics.append(
            fold_metrics
        )

        all_predictions.append(
            predictions
        )

        latest_forecasts.append(
            final_result[
                "forecast"
            ]
        )

        summary_rows.append(
            {
                "series_type":
                    "category",
                "series_id":
                    series_id,
                "model":
                    "xgboost",
                **aggregate_metrics,
            }
        )

        print(
            f"    Overall MAPE: "
            f"{aggregate_metrics['mean_mape']:.2f}%"
        )

    # ------------------------------------------------------------------
    # Regions
    # ------------------------------------------------------------------

    print(
        "\n=== REGION XGBOOST MODELS ==="
    )

    for index, series_id in enumerate(
        regions,
        start=1,
    ):

        print(
            f"\n[{index}/{len(regions)}] "
            f"Region: {series_id}"
        )

        series_df = get_single_series(
            features=features,
            series_type="region",
            series_id=series_id,
        )

        (
            fold_metrics,
            predictions,
        ) = evaluate_series(
            series_type="region",
            series_id=series_id,
            series_df=series_df,
        )

        aggregate_metrics = (
            aggregate_fold_metrics(
                fold_metrics
            )
        )

        final_result = (
            train_final_series_model(
                series_type="region",
                series_id=series_id,
                series_df=series_df,
            )
        )

        log_series_mlflow(
            series_type="region",
            series_id=series_id,
            aggregate_metrics=aggregate_metrics,
            fold_metrics=fold_metrics,
            model_path=final_result[
                "model_path"
            ],
        )

        all_fold_metrics.append(
            fold_metrics
        )

        all_predictions.append(
            predictions
        )

        latest_forecasts.append(
            final_result[
                "forecast"
            ]
        )

        summary_rows.append(
            {
                "series_type":
                    "region",
                "series_id":
                    series_id,
                "model":
                    "xgboost",
                **aggregate_metrics,
            }
        )

        print(
            f"    Overall MAPE: "
            f"{aggregate_metrics['mean_mape']:.2f}%"
        )

    # ------------------------------------------------------------------
    # Consolidate outputs
    # ------------------------------------------------------------------

    fold_metrics_df = pd.concat(
        all_fold_metrics,
        ignore_index=True,
    )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    latest_forecasts_df = pd.concat(
        latest_forecasts,
        ignore_index=True,
    )

    # ------------------------------------------------------------------
    # Sort summary
    # ------------------------------------------------------------------

    summary_df = (
        summary_df.sort_values(
            [
                "series_type",
                "mean_mape",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # Rank within each series type.
    summary_df["rank_within_type"] = (
        summary_df.groupby(
            "series_type"
        )["mean_mape"]
        .rank(
            method="dense",
            ascending=True,
        )
        .astype(int)
    )

    # ------------------------------------------------------------------
    # Save reports
    # ------------------------------------------------------------------

    fold_metrics_path = (
        REPORT_DIR
        / "secondary_fold_metrics.csv"
    )

    summary_path = (
        REPORT_DIR
        / "secondary_model_summary.csv"
    )

    predictions_path = (
        REPORT_DIR
        / "secondary_predictions.parquet"
    )

    latest_forecasts_path = (
        REPORT_DIR
        / "secondary_latest_forecasts.csv"
    )

    fold_metrics_df.to_csv(
        fold_metrics_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    predictions_df.to_parquet(
        predictions_path,
        index=False,
    )

    latest_forecasts_df.to_csv(
        latest_forecasts_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # Final validation / console summary
    # ------------------------------------------------------------------

    print(
        "\n=== SECONDARY ML COMPLETE ==="
    )

    print(
        f"\nTotal category models: "
        f"{len(categories)}"
    )

    print(
        f"Total region models: "
        f"{len(regions)}"
    )

    print(
        f"Total XGBoost models: "
        f"{len(summary_df)}"
    )

    print(
        "\n=== CATEGORY SUMMARY ==="
    )

    print(
        summary_df.loc[
            summary_df["series_type"]
            == "category"
        ]
        [
            [
                "series_id",
                "mean_mape",
                "mean_rmse",
                "mean_mae",
                "mean_interval_coverage_pct",
                "rank_within_type",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\n=== REGION SUMMARY ==="
    )

    print(
        summary_df.loc[
            summary_df["series_type"]
            == "region"
        ]
        [
            [
                "series_id",
                "mean_mape",
                "mean_rmse",
                "mean_mae",
                "mean_interval_coverage_pct",
                "rank_within_type",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\n=== OUTPUTS ==="
    )

    print(
        f"Fold metrics:\n"
        f"{fold_metrics_path}"
    )

    print(
        f"\nModel summary:\n"
        f"{summary_path}"
    )

    print(
        f"\nOOF predictions:\n"
        f"{predictions_path}"
    )

    print(
        f"\nLatest forecasts:\n"
        f"{latest_forecasts_path}"
    )

    print(
        f"\nModel artifacts:\n"
        f"{MODEL_DIR}"
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_secondary_training()