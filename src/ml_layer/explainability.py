"""
SHAP Explainability Layer
=========================

Production explainability for all 21 XGBoost forecasting series.

Production model decision
--------------------------
XGBoost is the production model for:
    - Overall revenue
    - 15 business categories
    - 5 regions

LSTM remains a benchmark in the model comparison, but is not
productionized because consistent TreeSHAP explanations are more
important for the end-to-end agent architecture.

This module produces:

1. Global SHAP feature importance
2. Local SHAP explanations for the next 4 forecasts
3. Machine-readable explanation files
4. Human-readable top UP / DOWN drivers

Outputs
-------
reports/shap/
    global_feature_importance.csv
    overall_local_explanations.csv
    secondary_local_explanations.csv
    shap_summary.csv

The local explanations are designed to be consumed later by
the AI / agent layer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor

from src.data_layer.feature_engineering import (
    build_recursive_feature_row,
)
from src.ml_layer.training import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    fit_xgb,
    recursive_xgb_forecast,
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

SECONDARY_MODEL_DIR = (
    MODEL_DIR
    / "secondary"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "shap"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

FORECAST_HORIZON = 4

TOP_DRIVERS = 5


# ============================================================================
# DIRECTORY SETUP
# ============================================================================

def ensure_directories() -> None:
    """Create the SHAP report directory."""
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================================
# DATA LOADING
# ============================================================================

def load_feature_dataset() -> pd.DataFrame:
    """Load the canonical forecasting feature dataset."""

    if not FEATURE_DATASET_PATH.is_file():
        raise FileNotFoundError(
            "Feature dataset not found:\n"
            f"{FEATURE_DATASET_PATH}"
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
            f"columns: {sorted(missing_columns)}"
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
# HELPERS
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


def get_series(
    features: pd.DataFrame,
    series_type: str,
    series_id: str,
) -> pd.DataFrame:
    """Return one complete series."""

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

    return (
        series.sort_values(
            "timestamp"
        )
        .reset_index(drop=True)
    )


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_xgb_model(
    model_path: Path,
) -> XGBRegressor:
    """Load a saved XGBoost model artifact."""

    if not model_path.is_file():
        raise FileNotFoundError(
            f"XGBoost model artifact not found:\n"
            f"{model_path}"
        )

    model = XGBRegressor()

    model.load_model(
        model_path
    )

    return model


# ============================================================================
# OVERALL PRODUCTION MODEL
# ============================================================================

def train_and_save_overall_production_model(
    overall: pd.DataFrame,
) -> tuple[XGBRegressor, Path]:
    """
    Train the final production XGBoost model for overall revenue
    using the complete approved modeling history.

    This is NOT a validation run. The validation comparison has
    already been completed in training.py.
    """

    model = fit_xgb(
        overall
    )

    model_path = (
        MODEL_DIR
        / "overall_production_xgboost.json"
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_model(
        model_path
    )

    return (
        model,
        model_path,
    )


# ============================================================================
# TREE SHAP
# ============================================================================

def create_tree_explainer(
    model: XGBRegressor,
) -> shap.TreeExplainer:
    """
    Create the TreeSHAP explainer.

    TreeSHAP is used consistently across all production XGBoost
    models.
    """

    return shap.TreeExplainer(
        model
    )


# ============================================================================
# GLOBAL SHAP
# ============================================================================

def calculate_global_shap(
    model: XGBRegressor,
    series_df: pd.DataFrame,
    series_type: str,
    series_id: str,
) -> pd.DataFrame:
    """
    Calculate global SHAP feature importance.

    Importance is mean absolute SHAP value across the supplied
    historical feature rows.
    """

    X = series_df[
        FEATURE_COLUMNS
    ].copy()

    explainer = create_tree_explainer(
        model
    )

    shap_values = explainer.shap_values(
        X
    )

    shap_values = np.asarray(
        shap_values
    )

    if shap_values.ndim == 3:
        shap_values = shap_values[0]

    mean_abs_shap = np.mean(
        np.abs(
            shap_values
        ),
        axis=0,
    )

    importance = pd.DataFrame(
        {
            "series_type":
                series_type,
            "series_id":
                series_id,
            "feature":
                FEATURE_COLUMNS,
            "mean_abs_shap":
                mean_abs_shap,
        }
    )

    total_importance = (
        importance[
            "mean_abs_shap"
        ].sum()
    )

    if total_importance > 0:
        importance[
            "importance_pct"
        ] = (
            importance[
                "mean_abs_shap"
            ]
            / total_importance
            * 100.0
        )
    else:
        importance[
            "importance_pct"
        ] = 0.0

    importance = (
        importance.sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    importance[
        "global_rank"
    ] = (
        np.arange(
            len(importance)
        )
        + 1
    )

    return importance


# ============================================================================
# BUILD FUTURE FEATURE ROWS
# ============================================================================

def build_future_feature_rows(
    series_df: pd.DataFrame,
    horizon: int = FORECAST_HORIZON,
) -> pd.DataFrame:
    """
    Build the feature rows for the next forecast horizon.

    Predictions are recursively fed into the history so the
    future features match deployment behavior.
    """

    history = pd.Series(
        series_df[
            TARGET_COLUMN
        ].to_numpy(
            dtype=float
        ),
        index=pd.DatetimeIndex(
            series_df[
                "timestamp"
            ]
        ),
        dtype="float64",
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
        periods=horizon,
        freq="W-SUN",
    )

    rows: list[dict] = []

    for timestamp in future_timestamps:

        feature_row = (
            build_recursive_feature_row(
                history=history,
                timestamp=timestamp,
            )
        )

        rows.append(
            {
                "timestamp":
                    timestamp,
                **{
                    feature:
                        float(
                            feature_row.iloc[
                                0
                            ][feature]
                        )
                    for feature
                    in FEATURE_COLUMNS
                },
            }
        )

        # We need the model prediction before adding it
        # to history. This function only creates the feature
        # structure, so prediction is handled separately.

        # Temporarily mark the timestamp. The caller will
        # reconstruct the recursive history using predictions.

    return pd.DataFrame(
        rows
    )


def build_recursive_future_inputs(
    model: XGBRegressor,
    series_df: pd.DataFrame,
    horizon: int = FORECAST_HORIZON,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
]:
    """
    Build recursive future feature rows and predictions.

    Returns
    -------
    future_features:
        Feature matrix used for the four forecasts.

    predictions:
        Corresponding recursive XGBoost predictions.
    """

    history = pd.Series(
        series_df[
            TARGET_COLUMN
        ].to_numpy(
            dtype=float
        ),
        index=pd.DatetimeIndex(
            series_df[
                "timestamp"
            ]
        ),
        dtype="float64",
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
        periods=horizon,
        freq="W-SUN",
    )

    feature_rows: list[pd.Series] = []
    predictions: list[float] = []

    model.eval if hasattr(model, "eval") else None

    for timestamp in future_timestamps:

        feature_row = (
            build_recursive_feature_row(
                history=history,
                timestamp=timestamp,
            )
        )

        feature_rows.append(
            feature_row[
                FEATURE_COLUMNS
            ].iloc[0]
        )

        prediction = float(
            model.predict(
                feature_row[
                    FEATURE_COLUMNS
                ]
            )[0]
        )

        prediction = max(
            prediction,
            0.0,
        )

        predictions.append(
            prediction
        )

        history.loc[
            timestamp
        ] = prediction

    future_features = pd.DataFrame(
        feature_rows
    ).reset_index(
        drop=True
    )

    return (
        future_features,
        np.asarray(
            predictions,
            dtype=float,
        ),
    )


# ============================================================================
# LOCAL SHAP
# ============================================================================

def calculate_local_shap(
    model: XGBRegressor,
    future_features: pd.DataFrame,
    predictions: np.ndarray,
    series_type: str,
    series_id: str,
    timestamps: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Calculate local TreeSHAP explanations for each future forecast.

    Positive SHAP contribution:
        pushes forecast upward.

    Negative SHAP contribution:
        pushes forecast downward.
    """

    explainer = create_tree_explainer(
        model
    )

    shap_values = explainer.shap_values(
        future_features[
            FEATURE_COLUMNS
        ]
    )

    shap_values = np.asarray(
        shap_values
    )

    if shap_values.ndim == 3:
        shap_values = shap_values[0]

    rows: list[dict] = []

    base_value = float(
        np.asarray(
            explainer.expected_value
        ).reshape(-1)[0]
    )

    for row_index, timestamp in enumerate(
        timestamps
    ):

        feature_values = future_features.iloc[
            row_index
        ]

        contributions = shap_values[
            row_index
        ]

        for feature_index, feature in enumerate(
            FEATURE_COLUMNS
        ):

            contribution = float(
                contributions[
                    feature_index
                ]
            )

            rows.append(
                {
                    "series_type":
                        series_type,
                    "series_id":
                        series_id,
                    "forecast_timestamp":
                        timestamp,
                    "forecast_revenue":
                        float(
                            predictions[
                                row_index
                            ]
                        ),
                    "feature":
                        feature,
                    "feature_value":
                        float(
                            feature_values[
                                feature
                            ]
                        ),
                    "shap_value":
                        contribution,
                    "direction":
                        (
                            "up"
                            if contribution > 0
                            else (
                                "down"
                                if contribution < 0
                                else "neutral"
                            )
                        ),
                    "base_value":
                        base_value,
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# TOP LOCAL DRIVERS
# ============================================================================

def build_top_driver_summary(
    local_shap: pd.DataFrame,
    top_n: int = TOP_DRIVERS,
) -> pd.DataFrame:
    """
    Create a compact local explanation summary with top features
    pushing the forecast UP and DOWN.
    """

    rows: list[dict] = []

    group_columns = [
        "series_type",
        "series_id",
        "forecast_timestamp",
        "forecast_revenue",
        "base_value",
    ]

    for group_key, group in local_shap.groupby(
        group_columns,
        sort=False,
    ):

        (
            series_type,
            series_id,
            timestamp,
            forecast_revenue,
            base_value,
        ) = group_key

        upward = (
            group[
                group["shap_value"] > 0
            ]
            .sort_values(
                "shap_value",
                ascending=False,
            )
            .head(top_n)
        )

        downward = (
            group[
                group["shap_value"] < 0
            ]
            .sort_values(
                "shap_value",
                ascending=True,
            )
            .head(top_n)
        )

        row = {
            "series_type":
                series_type,
            "series_id":
                series_id,
            "forecast_timestamp":
                timestamp,
            "forecast_revenue":
                forecast_revenue,
            "base_value":
                base_value,
        }

        for rank in range(
            1,
            top_n + 1,
        ):

            if rank <= len(upward):
                item = upward.iloc[
                    rank - 1
                ]

                row[
                    f"up_{rank}_feature"
                ] = item[
                    "feature"
                ]

                row[
                    f"up_{rank}_shap"
                ] = item[
                    "shap_value"
                ]

            else:
                row[
                    f"up_{rank}_feature"
                ] = None

                row[
                    f"up_{rank}_shap"
                ] = None

            if rank <= len(downward):
                item = downward.iloc[
                    rank - 1
                ]

                row[
                    f"down_{rank}_feature"
                ] = item[
                    "feature"
                ]

                row[
                    f"down_{rank}_shap"
                ] = item[
                    "shap_value"
                ]

            else:
                row[
                    f"down_{rank}_feature"
                ] = None

                row[
                    f"down_{rank}_shap"
                ] = None

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# PROCESS ONE SERIES
# ============================================================================

def explain_series(
    features: pd.DataFrame,
    series_type: str,
    series_id: str,
    model: XGBRegressor,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Generate global and local SHAP explanations for one series."""

    series_df = get_series(
        features,
        series_type,
        series_id,
    )

    # ------------------------------------------------------------------
    # Global explanation
    # ------------------------------------------------------------------

    global_importance = calculate_global_shap(
        model=model,
        series_df=series_df,
        series_type=series_type,
        series_id=series_id,
    )

    # ------------------------------------------------------------------
    # Local future explanation
    # ------------------------------------------------------------------

    (
        future_features,
        predictions,
    ) = build_recursive_future_inputs(
        model=model,
        series_df=series_df,
        horizon=FORECAST_HORIZON,
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
        periods=FORECAST_HORIZON,
        freq="W-SUN",
    )

    local_importance = calculate_local_shap(
        model=model,
        future_features=future_features,
        predictions=predictions,
        series_type=series_type,
        series_id=series_id,
        timestamps=future_timestamps,
    )

    return (
        global_importance,
        local_importance,
    )


# ============================================================================
# MAIN EXPLAINABILITY PIPELINE
# ============================================================================

def run_explainability() -> None:
    """Generate SHAP explanations for all 21 production series."""

    print(
        "=== SHAP EXPLAINABILITY PIPELINE ==="
    )

    ensure_directories()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------

    print(
        "\nLoading feature dataset..."
    )

    features = load_feature_dataset()

    print(
        f"Loaded {len(features):,} rows."
    )

    # ------------------------------------------------------------------
    # Overall production model
    # ------------------------------------------------------------------

    print(
        "\n=== OVERALL PRODUCTION XGBOOST ==="
    )

    overall = get_series(
        features,
        "overall",
        "overall",
    )

    overall_model, overall_model_path = (
        train_and_save_overall_production_model(
            overall
        )
    )

    print(
        "Saved overall production model:"
    )

    print(
        overall_model_path
    )

    # ------------------------------------------------------------------
    # Containers
    # ------------------------------------------------------------------

    global_results: list[pd.DataFrame] = []
    local_results: list[pd.DataFrame] = []

    # ------------------------------------------------------------------
    # Overall
    # ------------------------------------------------------------------

    print(
        "\nExplaining overall revenue..."
    )

    (
        overall_global,
        overall_local,
    ) = explain_series(
        features=features,
        series_type="overall",
        series_id="overall",
        model=overall_model,
    )

    global_results.append(
        overall_global
    )

    local_results.append(
        overall_local
    )

    print(
        "Overall SHAP completed."
    )

    # ------------------------------------------------------------------
    # Secondary models
    # ------------------------------------------------------------------

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
        "\n=== SECONDARY MODEL EXPLANATIONS ==="
    )

    for _, row in secondary_series.iterrows():

        series_type = str(
            row["series_type"]
        )

        series_id = str(
            row["series_id"]
        )

        model_filename = (
            f"{slugify(series_type)}_"
            f"{slugify(series_id)}.json"
        )

        model_path = (
            SECONDARY_MODEL_DIR
            / model_filename
        )

        print(
            f"\nExplaining "
            f"{series_type}: "
            f"{series_id}"
        )

        model = load_xgb_model(
            model_path
        )

        (
            global_importance,
            local_importance,
        ) = explain_series(
            features=features,
            series_type=series_type,
            series_id=series_id,
            model=model,
        )

        global_results.append(
            global_importance
        )

        local_results.append(
            local_importance
        )

        print(
            "SHAP completed."
        )

    # ------------------------------------------------------------------
    # Combine results
    # ------------------------------------------------------------------

    global_df = pd.concat(
        global_results,
        ignore_index=True,
    )

    local_df = pd.concat(
        local_results,
        ignore_index=True,
    )

    # ------------------------------------------------------------------
    # Top-driver summary
    # ------------------------------------------------------------------

    local_summary = (
        build_top_driver_summary(
            local_df,
            top_n=TOP_DRIVERS,
        )
    )

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    summary_rows: list[dict] = []

    for (
        series_type,
        series_id,
    ), group in global_df.groupby(
        [
            "series_type",
            "series_id",
        ],
        sort=False,
    ):

        top_feature = (
            group
            .sort_values(
                "mean_abs_shap",
                ascending=False,
            )
            .iloc[0]
        )

        summary_rows.append(
            {
                "series_type":
                    series_type,
                "series_id":
                    series_id,
                "top_global_feature":
                    top_feature[
                        "feature"
                    ],
                "top_global_mean_abs_shap":
                    float(
                        top_feature[
                            "mean_abs_shap"
                        ]
                    ),
                "top_global_importance_pct":
                    float(
                        top_feature[
                            "importance_pct"
                        ]
                    ),
            }
        )

    shap_summary = pd.DataFrame(
        summary_rows
    )

    # ------------------------------------------------------------------
    # Save machine-readable artifacts
    # ------------------------------------------------------------------

    global_path = (
        REPORT_DIR
        / "global_feature_importance.csv"
    )

    overall_local_path = (
        REPORT_DIR
        / "overall_local_explanations.csv"
    )

    secondary_local_path = (
        REPORT_DIR
        / "secondary_local_explanations.csv"
    )

    summary_path = (
        REPORT_DIR
        / "shap_summary.csv"
    )

    local_summary_path = (
        REPORT_DIR
        / "top_forecast_drivers.csv"
    )

    global_df.to_csv(
        global_path,
        index=False,
    )

    overall_local = local_df[
        (
            local_df["series_type"]
            == "overall"
        )
    ].copy()

    secondary_local = local_df[
        (
            local_df["series_type"]
            != "overall"
        )
    ].copy()

    overall_local.to_csv(
        overall_local_path,
        index=False,
    )

    secondary_local.to_csv(
        secondary_local_path,
        index=False,
    )

    shap_summary.to_csv(
        summary_path,
        index=False,
    )

    local_summary.to_csv(
        local_summary_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # Create compact JSON for later AI-agent consumption
    # ------------------------------------------------------------------

    agent_explanations: list[dict] = []

    for (
        series_type,
        series_id,
        timestamp,
    ), group in local_df.groupby(
        [
            "series_type",
            "series_id",
            "forecast_timestamp",
        ],
        sort=False,
    ):

        group = group.sort_values(
            "shap_value",
            ascending=False,
        )

        top_up = (
            group[
                group["shap_value"] > 0
            ]
            .head(TOP_DRIVERS)
        )

        top_down = (
            group[
                group["shap_value"] < 0
            ]
            .sort_values(
                "shap_value",
                ascending=True,
            )
            .head(TOP_DRIVERS)
        )

        agent_explanations.append(
            {
                "series_type":
                    series_type,
                "series_id":
                    series_id,
                "forecast_timestamp":
                    str(timestamp),
                "forecast_revenue":
                    float(
                        group[
                            "forecast_revenue"
                        ].iloc[0]
                    ),
                "base_value":
                    float(
                        group[
                            "base_value"
                        ].iloc[0]
                    ),
                "drivers_up":
                    [
                        {
                            "feature":
                                row["feature"],
                            "feature_value":
                                float(
                                    row[
                                        "feature_value"
                                    ]
                                ),
                            "shap_value":
                                float(
                                    row[
                                        "shap_value"
                                    ]
                                ),
                        }
                        for _, row
                        in top_up.iterrows()
                    ],
                "drivers_down":
                    [
                        {
                            "feature":
                                row["feature"],
                            "feature_value":
                                float(
                                    row[
                                        "feature_value"
                                    ]
                                ),
                            "shap_value":
                                float(
                                    row[
                                        "shap_value"
                                    ]
                                ),
                        }
                        for _, row
                        in top_down.iterrows()
                    ],
            }
        )

    agent_json_path = (
        REPORT_DIR
        / "agent_shap_explanations.json"
    )

    with open(
        agent_json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            agent_explanations,
            file,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Final console summary
    # ------------------------------------------------------------------

    print(
        "\n=== SHAP PIPELINE COMPLETE ==="
    )

    print(
        "\nExplained series:"
    )

    category_count = int(
        (
            secondary_series["series_type"]
            == "category"
        ).sum()
    )

    region_count = int(
        (
            secondary_series["series_type"]
            == "region"
        ).sum()
    )

    print(
        "Overall: 1"
    )

    print(
        f"Categories: {category_count}"
    )

    print(
        f"Regions: {region_count}"
    )

    print(
        "\n=== OUTPUTS ==="
    )

    print(
        f"Global SHAP:\n"
        f"{global_path}"
    )

    print(
        f"\nOverall local SHAP:\n"
        f"{overall_local_path}"
    )

    print(
        f"\nSecondary local SHAP:\n"
        f"{secondary_local_path}"
    )

    print(
        f"\nSHAP summary:\n"
        f"{summary_path}"
    )

    print(
        f"\nTop forecast drivers:\n"
        f"{local_summary_path}"
    )

    print(
        f"\nAgent-ready explanations:\n"
        f"{agent_json_path}"
    )


if __name__ == "__main__":
    run_explainability()