"""
Stage 1 ML training pipeline for the Intelligent Business
Forecasting Agent.

Implemented in this stage
-------------------------
Overall revenue series only:

1. Exponential Smoothing baseline
2. XGBoost
3. 52-week expanding-window walk-forward validation
4. 4-week validation horizon
5. 4-week step
6. True recursive XGBoost forecasting
7. MAPE / RMSE / MAE
8. 80% prediction-interval coverage
9. MLflow experiment tracking

Not implemented yet
-------------------
- Prophet
- LSTM
- Weighted ensemble
- 15 category XGBoost models
- 5 region XGBoost models
- SHAP
- drift monitoring
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import random
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor

def set_random_seed(seed: int = 42) -> None:
    """Make model training as reproducible as practical."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from src.data_layer.cleaning import (
    MODEL_END,
    MODEL_START,
)
from src.data_layer.feature_engineering import (
    build_recursive_feature_row,
)
from src.ml_layer.evaluation import (
    aggregate_fold_metrics,
    calculate_metrics,
    generate_walk_forward_folds,
)


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT: Final[Path] = (
    Path(__file__).resolve().parents[2]
)

FEATURE_DATASET_PATH: Final[Path] = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "forecasting_features.parquet"
)

MODEL_DIR: Final[Path] = (
    PROJECT_ROOT
    / "src"
    / "ml_layer"
    / "models"
)

REPORT_DIR: Final[Path] = (
    PROJECT_ROOT
    / "reports"
)

MLRUNS_DIR: Final[Path] = (
    PROJECT_ROOT
    / "mlruns"
)

STAGE1_REPORT_PATH: Final[Path] = (
    REPORT_DIR
    / "stage1_overall_model_comparison.csv"
)

XGB_FOLD_REPORT_PATH: Final[Path] = (
    REPORT_DIR
    / "stage1_xgboost_fold_metrics.csv"
)

ETS_FOLD_REPORT_PATH: Final[Path] = (
    REPORT_DIR
    / "stage1_exponential_smoothing_fold_metrics.csv"
)


# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================

FEATURE_COLUMNS: Final[list[str]] = [
    "lag_1",
    "lag_4",
    "lag_52",
    "rolling_mean_4",
    "rolling_std_4",
    "year",
    "month",
    "quarter",
    "week_of_year",
    "day_of_week",
    "is_weekend",
    "days_to_month_end",
    "month_sin",
    "month_cos",
    "week_sin",
    "week_cos",
]

TARGET_COLUMN: Final[str] = "target_revenue"


# ============================================================================
# XGBOOST CONFIGURATION
# ============================================================================

XGB_PARAMS: Final[dict[str, object]] = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}

# ============================================================================
# LSTM CONFIGURATION
# ============================================================================

LSTM_SEQUENCE_LENGTH = 12
LSTM_HIDDEN_SIZE = 32
LSTM_EPOCHS = 150
LSTM_LEARNING_RATE = 0.01

# ============================================================================
# MODEL INTERVAL CONFIGURATION
# ============================================================================

LOWER_QUANTILE = 0.10
UPPER_QUANTILE = 0.90


# ============================================================================
# DIRECTORIES
# ============================================================================

def ensure_directories() -> None:
    """Create required ML output directories."""
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MLRUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================================
# MLflow
# ============================================================================

def configure_mlflow() -> None:
    """Configure local MLflow tracking using SQLite."""
    mlflow_db_path = (
        PROJECT_ROOT / "mlflow.db"
    )

    tracking_uri = (
        f"sqlite:///{mlflow_db_path.as_posix()}"
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    mlflow.set_experiment(
        "intelligent-forecasting-agent"
    )

    print(
        f"MLflow tracking URI: {tracking_uri}"
    )


# ============================================================================
# DATA LOADING
# ============================================================================

def load_feature_dataset() -> pd.DataFrame:
    """
    Load the canonical long-format forecasting feature dataset.

    The ML layer does not regenerate features.
    """
    if not FEATURE_DATASET_PATH.is_file():
        raise FileNotFoundError(
            "Feature dataset not found:\n"
            f"{FEATURE_DATASET_PATH}\n\n"
            "Run the Data Layer pipeline first."
        )

    features = pd.read_parquet(
        FEATURE_DATASET_PATH
    )

    features["timestamp"] = pd.to_datetime(
        features["timestamp"]
    )

    return features.sort_values(
        [
            "series_type",
            "series_id",
            "timestamp",
        ]
    ).reset_index(
        drop=True
    )


# ============================================================================
# OVERALL SERIES
# ============================================================================

def get_overall_series(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Return the overall revenue series only."""
    overall = features[
        (features["series_type"] == "overall")
        & (features["series_id"] == "overall")
    ].copy()

    overall = overall.sort_values(
        "timestamp"
    ).reset_index(
        drop=True
    )

    if overall.empty:
        raise ValueError(
            "Overall revenue series was not found."
        )

    expected_periods = 87

    if len(overall) != expected_periods:
        raise ValueError(
            "Expected "
            f"{expected_periods} overall weekly rows, "
            f"found {len(overall)}."
        )

    return overall


# ============================================================================
# METRIC / INTERVAL HELPERS
# ============================================================================

def build_residual_interval(
    calibration_actual: np.ndarray,
    calibration_predicted: np.ndarray,
    future_predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build an empirical 80% prediction interval from multiple
    out-of-sample calibration residuals.

    Calibration residuals must come from pseudo-walk-forward
    forecasts generated strictly inside the training window.
    """

    residuals = (
        calibration_actual
        - calibration_predicted
    )

    lower_residual = np.quantile(
        residuals,
        LOWER_QUANTILE,
    )

    upper_residual = np.quantile(
        residuals,
        UPPER_QUANTILE,
    )

    lower = np.maximum(
        future_predictions + lower_residual,
        0.0,
    )

    upper = np.maximum(
        future_predictions + upper_residual,
        0.0,
    )

    return lower, upper


def calculate_fold_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, float]:
    """Calculate the approved evaluation metrics."""
    return calculate_metrics(
        actual=actual,
        predicted=predicted,
        lower=lower,
        upper=upper,
    )

# ============================================================================
# LSTM MODEL
# ============================================================================

class RevenueLSTM(nn.Module):
    """Small univariate LSTM for weekly revenue forecasting."""

    def __init__(
        self,
        hidden_size: int = LSTM_HIDDEN_SIZE,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.output_layer = nn.Linear(
            hidden_size,
            1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        output, _ = self.lstm(x)

        last_output = output[:, -1, :]

        return self.output_layer(
            last_output
        )

def create_lstm_sequences(
    values: np.ndarray,
    sequence_length: int = LSTM_SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a univariate revenue series into supervised
    sequence/target pairs.
    """

    X: list[np.ndarray] = []
    y: list[float] = []

    for index in range(
        sequence_length,
        len(values),
    ):
        X.append(
            values[
                index - sequence_length:index
            ]
        )

        y.append(
            values[index]
        )

    if not X:
        raise ValueError(
            "Not enough observations to construct "
            "LSTM sequences."
        )

    return (
        np.asarray(X, dtype=float),
        np.asarray(y, dtype=float),
    )

def fit_lstm(
    train_values: np.ndarray,
) -> tuple[RevenueLSTM, StandardScaler]:
    """
    Fit the LSTM on the training portion only.

    Scaling is fitted only on training data to prevent leakage.
    """

    scaler = StandardScaler()

    scaled_values = (
        scaler.fit_transform(
            train_values.reshape(-1, 1)
        )
        .flatten()
    )

    X, y = create_lstm_sequences(
        scaled_values
    )

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
    ).unsqueeze(-1)

    y_tensor = torch.tensor(
        y,
        dtype=torch.float32,
    ).unsqueeze(-1)

    model = RevenueLSTM()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LSTM_LEARNING_RATE,
    )

    loss_function = nn.MSELoss()

    model.train()

    for _ in range(
        LSTM_EPOCHS
    ):
        optimizer.zero_grad()

        predictions = model(
            X_tensor
        )

        loss = loss_function(
            predictions,
            y_tensor,
        )

        loss.backward()

        optimizer.step()

    return (
        model,
        scaler,
    )

def recursive_lstm_forecast(
    model: RevenueLSTM,
    scaler: StandardScaler,
    history_values: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """
    Produce a true recursive multi-step forecast.

    Each prediction is appended to the history and becomes
    part of the input sequence for the next prediction.
    """

    history = list(
        history_values.astype(float)
    )

    predictions: list[float] = []

    model.eval()

    for _ in range(horizon):

        recent_values = np.asarray(
            history[
                -LSTM_SEQUENCE_LENGTH:
            ],
            dtype=float,
        )

        scaled_recent = scaler.transform(
            recent_values.reshape(-1, 1)
        )

        X_tensor = torch.tensor(
            scaled_recent,
            dtype=torch.float32,
        ).unsqueeze(0)

        with torch.no_grad():
            scaled_prediction = (
                model(
                    X_tensor
                )
                .item()
            )

        prediction = float(
            scaler.inverse_transform(
                np.array(
                    [[scaled_prediction]]
                )
            )[0, 0]
        )

        prediction = max(
            prediction,
            0.0,
        )

        predictions.append(
            prediction
        )

        history.append(
            prediction
        )

    return np.asarray(
        predictions,
        dtype=float,
    )

def calibrate_lstm_interval(
    train_df: pd.DataFrame,
    initial_calibration_weeks: int = 20,
    calibration_horizon: int = 4,
    calibration_step: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate out-of-sample LSTM residuals using nested
    rolling-origin calibration inside the outer training fold.
    """

    train_df = train_df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    calibration_actuals: list[float] = []
    calibration_predictions: list[float] = []

    train_size = len(train_df)

    origin = initial_calibration_weeks

    while (
        origin + calibration_horizon
        <= train_size
    ):
        pseudo_train = train_df.iloc[
            :origin
        ].copy()

        pseudo_validation = train_df.iloc[
            origin:
            origin + calibration_horizon
        ].copy()

        train_values = pseudo_train[
            TARGET_COLUMN
        ].to_numpy(
            dtype=float
        )

        model, scaler = fit_lstm(
            train_values
        )

        predictions = recursive_lstm_forecast(
            model=model,
            scaler=scaler,
            history_values=train_values,
            horizon=calibration_horizon,
        )

        actual = pseudo_validation[
            TARGET_COLUMN
        ].to_numpy(
            dtype=float
        )

        calibration_actuals.extend(
            actual.tolist()
        )

        calibration_predictions.extend(
            predictions.tolist()
        )

        origin += calibration_step

    if not calibration_actuals:
        raise ValueError(
            "Unable to generate LSTM calibration "
            "residuals."
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

def evaluate_lstm_fold(
    fold_number: int,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> dict[str, object]:
    """Train and evaluate LSTM on one walk-forward fold."""

    train_values = train_df[
        TARGET_COLUMN
    ].to_numpy(
        dtype=float
    )

    actual = validation_df[
        TARGET_COLUMN
    ].to_numpy(
        dtype=float
    )

    model, scaler = fit_lstm(
        train_values
    )

    predicted = recursive_lstm_forecast(
        model=model,
        scaler=scaler,
        history_values=train_values,
        horizon=len(validation_df),
    )

    calibration_actual, calibration_predicted = (
        calibrate_lstm_interval(
            train_df,
            initial_calibration_weeks=20,
            calibration_horizon=4,
            calibration_step=4,
        )
    )

    lower, upper = build_residual_interval(
        calibration_actual=calibration_actual,
        calibration_predicted=calibration_predicted,
        future_predictions=predicted,
    )

    metrics = calculate_fold_metrics(
        actual=actual,
        predicted=predicted,
        lower=lower,
        upper=upper,
    )

    return {
        "fold": fold_number,
        "actual": actual,
        "predicted": predicted,
        "lower": lower,
        "upper": upper,
        "metrics": metrics,
        "model": model,
        "scaler": scaler,
    }

def walk_forward_lstm(
    overall: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the approved 8-fold expanding-window LSTM evaluation.
    """

    folds = generate_walk_forward_folds(
        timestamps=overall[
            "timestamp"
        ]
    )

    fold_metrics: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for fold in folds:

        train_df = overall[
            overall["timestamp"]
            <= fold.train_end
        ].copy()

        validation_df = overall[
            (
                overall["timestamp"]
                >= fold.validation_start
            )
            & (
                overall["timestamp"]
                <= fold.validation_end
            )
        ].copy()

        if len(train_df) != (
            52
            + (fold.fold_number - 1) * 4
        ):
            raise ValueError(
                f"Unexpected training size "
                f"for LSTM fold "
                f"{fold.fold_number}: "
                f"{len(train_df)}"
            )

        if len(validation_df) != 4:
            raise ValueError(
                f"Unexpected validation size "
                f"for LSTM fold "
                f"{fold.fold_number}: "
                f"{len(validation_df)}"
            )

        result = evaluate_lstm_fold(
            fold_number=fold.fold_number,
            train_df=train_df,
            validation_df=validation_df,
        )

        metrics = result["metrics"]

        fold_metrics.append(
            {
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
            validation_df["timestamp"]
        ):
            prediction_rows.append(
                {
                    "fold":
                        fold.fold_number,
                    "timestamp":
                        timestamp,
                    "actual":
                        result["actual"][index],
                    "predicted":
                        result["predicted"][index],
                    "lower_80":
                        result["lower"][index],
                    "upper_80":
                        result["upper"][index],
                }
            )

        print(
            f"LSTM fold "
            f"{fold.fold_number}/{len(folds)} | "
            f"MAPE={metrics['mape']:.2f}% | "
            f"RMSE={metrics['rmse']:.2f} | "
            f"MAE={metrics['mae']:.2f}"
        )

    return (
        pd.DataFrame(fold_metrics),
        pd.DataFrame(prediction_rows),
    )


# ============================================================================
# XGBOOST
# ============================================================================

def fit_xgb(
    train_df: pd.DataFrame,
) -> XGBRegressor:
    """
    Train XGBoost on the canonical engineered features.

    XGBoost natively handles missing feature values, so the early
    lag_52 NaNs are intentionally retained.
    """
    model = XGBRegressor(
        **XGB_PARAMS,
    )

    X_train = train_df[
        FEATURE_COLUMNS
    ]

    y_train = train_df[
        TARGET_COLUMN
    ]

    model.fit(
        X_train,
        y_train,
    )

    return model

def calibrate_xgb_interval(
    train_df: pd.DataFrame,
    initial_calibration_weeks: int = 20,
    calibration_horizon: int = 4,
    calibration_step: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate multiple out-of-sample residuals using nested
    rolling-origin validation inside the outer training window.

    For a 52-week outer training fold, the default configuration
    creates several pseudo-forecast origins:

        20 weeks train -> 4 weeks forecast
        24 weeks train -> 4 weeks forecast
        28 weeks train -> 4 weeks forecast
        ...
        48 weeks train -> 4 weeks forecast

    These forecasts are never generated using the actual future
    value being predicted.
    """

    train_df = train_df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    calibration_actuals: list[float] = []
    calibration_predictions: list[float] = []

    train_size = len(train_df)

    origin = initial_calibration_weeks

    while (
        origin + calibration_horizon
        <= train_size
    ):
        pseudo_train = train_df.iloc[
            :origin
        ].copy()

        pseudo_validation = train_df.iloc[
            origin:
            origin + calibration_horizon
        ].copy()

        model = fit_xgb(
            pseudo_train
        )

        future_timestamps = pd.DatetimeIndex(
            pseudo_validation[
                "timestamp"
            ]
        )

        predictions = recursive_xgb_forecast(
            model=model,
            train_df=pseudo_train,
            future_timestamps=future_timestamps,
        )

        actual = pseudo_validation[
            TARGET_COLUMN
        ].to_numpy(
            dtype=float
        )

        calibration_actuals.extend(
            actual.tolist()
        )

        calibration_predictions.extend(
            predictions.tolist()
        )

        origin += calibration_step

    if not calibration_actuals:
        raise ValueError(
            "Unable to generate XGBoost calibration "
            "residuals from the training fold."
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


def recursive_xgb_forecast(
    model: XGBRegressor,
    train_df: pd.DataFrame,
    future_timestamps: pd.DatetimeIndex,
) -> np.ndarray:
    """
    Produce a true recursive multi-step forecast.

    The prediction for week t becomes part of the history used
    to construct features for week t+1.

    No actual future validation target is used.
    """
    history = pd.Series(
        train_df[TARGET_COLUMN].to_numpy(
            dtype=float
        ),
        index=pd.DatetimeIndex(
            train_df["timestamp"]
        ),
        dtype="float64",
    )

    predictions: list[float] = []

    for timestamp in future_timestamps:

        feature_row = build_recursive_feature_row(
            history=history,
            timestamp=pd.Timestamp(timestamp),
        )

        prediction = float(
            model.predict(
                feature_row[
                    FEATURE_COLUMNS
                ]
            )[0]
        )

        # Revenue cannot meaningfully be negative.
        prediction = max(
            prediction,
            0.0,
        )

        predictions.append(
            prediction
        )

        # Feed the prediction back into history.
        history.loc[
            timestamp
        ] = prediction

    return np.asarray(
        predictions,
        dtype=float,
    )


def evaluate_xgb_fold(
    fold_number: int,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> dict[str, object]:
    """Train and evaluate XGBoost on one validation fold."""

    model = fit_xgb(
        train_df
    )

    # ------------------------------------------------------------
    # Out-of-sample interval calibration
    # ------------------------------------------------------------

    calibration_actual, calibration_predicted = (
        calibrate_xgb_interval(
            train_df,
            initial_calibration_weeks=20,
            calibration_horizon=4,
            calibration_step=4,
        )
    )

    future_timestamps = pd.DatetimeIndex(
        validation_df[
            "timestamp"
        ]
    )

    predicted = recursive_xgb_forecast(
        model=model,
        train_df=train_df,
        future_timestamps=future_timestamps,
    )

    lower, upper = build_residual_interval(
        calibration_actual=calibration_actual,
        calibration_predicted=calibration_predicted,
        future_predictions=predicted,
    )

    actual = validation_df[
        TARGET_COLUMN
    ].to_numpy(
        dtype=float
    )

    metrics = calculate_fold_metrics(
        actual=actual,
        predicted=predicted,
        lower=lower,
        upper=upper,
    )

    return {
        "fold": fold_number,
        "actual": actual,
        "predicted": predicted,
        "lower": lower,
        "upper": upper,
        "metrics": metrics,
        "model": model,
    }


def walk_forward_xgb(
    overall: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the approved 8-fold recursive XGBoost evaluation.
    """
    folds = generate_walk_forward_folds(
        timestamps=overall[
            "timestamp"
        ]
    )

    fold_metrics: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for fold in folds:

        train_df = overall[
            overall["timestamp"]
            <= fold.train_end
        ].copy()

        validation_df = overall[
            (
                overall["timestamp"]
                >= fold.validation_start
            )
            & (
                overall["timestamp"]
                <= fold.validation_end
            )
        ].copy()

        if len(train_df) != (
            52
            + (fold.fold_number - 1) * 4
        ):
            raise ValueError(
                f"Unexpected training size "
                f"for fold {fold.fold_number}: "
                f"{len(train_df)}"
            )

        if len(validation_df) != 4:
            raise ValueError(
                f"Unexpected validation size "
                f"for fold {fold.fold_number}: "
                f"{len(validation_df)}"
            )

        result = evaluate_xgb_fold(
            fold_number=fold.fold_number,
            train_df=train_df,
            validation_df=validation_df,
        )

        metrics = result["metrics"]

        fold_metrics.append(
            {
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
            validation_df["timestamp"]
        ):
            prediction_rows.append(
                {
                    "fold":
                        fold.fold_number,
                    "timestamp":
                        timestamp,
                    "actual":
                        result["actual"][index],
                    "predicted":
                        result["predicted"][index],
                    "lower_80":
                        result["lower"][index],
                    "upper_80":
                        result["upper"][index],
                }
            )

        print(
            f"XGBoost fold {fold.fold_number}/"
            f"{len(folds)} | "
            f"MAPE={metrics['mape']:.2f}% | "
            f"RMSE={metrics['rmse']:.2f} | "
            f"MAE={metrics['mae']:.2f}"
        )

    return (
        pd.DataFrame(fold_metrics),
        pd.DataFrame(prediction_rows),
    )


# ============================================================================
# EXPONENTIAL SMOOTHING BASELINE
# ============================================================================
def calibrate_exponential_smoothing_interval(
    train_df: pd.DataFrame,
    initial_calibration_weeks: int = 20,
    calibration_horizon: int = 4,
    calibration_step: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate multiple out-of-sample residuals using nested
    rolling-origin validation for Exponential Smoothing.
    """

    train_df = train_df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    calibration_actuals: list[float] = []
    calibration_predictions: list[float] = []

    train_size = len(train_df)

    origin = initial_calibration_weeks

    while (
        origin + calibration_horizon
        <= train_size
    ):
        pseudo_train = train_df.iloc[
            :origin
        ].copy()

        pseudo_validation = train_df.iloc[
            origin:
            origin + calibration_horizon
        ].copy()

        model = ExponentialSmoothing(
            pseudo_train[
                TARGET_COLUMN
            ].to_numpy(dtype=float),
            trend="add",
            damped_trend=True,
            seasonal=None,
            initialization_method="estimated",
        ).fit(
            optimized=True
        )

        predictions = np.asarray(
            model.forecast(
                calibration_horizon
            ),
            dtype=float,
        )

        actual = pseudo_validation[
            TARGET_COLUMN
        ].to_numpy(
            dtype=float
        )

        calibration_actuals.extend(
            actual.tolist()
        )

        calibration_predictions.extend(
            predictions.tolist()
        )

        origin += calibration_step

    if not calibration_actuals:
        raise ValueError(
            "Unable to generate Exponential Smoothing "
            "calibration residuals from the training fold."
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


def evaluate_exponential_smoothing_fold(
    fold_number: int,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> dict[str, object]:
    """Evaluate one Exponential Smoothing fold."""

    train_values = train_df[
        TARGET_COLUMN
    ].to_numpy(
        dtype=float
    )

    actual = validation_df[
        TARGET_COLUMN
    ].to_numpy(
        dtype=float
    )

    model = ExponentialSmoothing(
        train_values,
        trend="add",
        damped_trend=True,
        seasonal=None,
        initialization_method="estimated",
    ).fit(
        optimized=True
    )

    predicted = np.asarray(
        model.forecast(
            len(validation_df)
        ),
        dtype=float,
    )

    predicted = np.maximum(
        predicted,
        0.0,
    )

    calibration_actual, calibration_predicted = (
        calibrate_exponential_smoothing_interval(
            train_df,
            initial_calibration_weeks=20,
            calibration_horizon=4,
            calibration_step=4,
        )
    )

    lower, upper = build_residual_interval(
        calibration_actual=calibration_actual,
        calibration_predicted=calibration_predicted,
        future_predictions=predicted,
    )

    metrics = calculate_fold_metrics(
        actual=actual,
        predicted=predicted,
        lower=lower,
        upper=upper,
    )

    return {
        "fold": fold_number,
        "metrics": metrics,
        "actual": actual,
        "predicted": predicted,
        "lower": lower,
        "upper": upper,
    }


def walk_forward_exponential_smoothing(
    overall: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the approved 8-fold statistical baseline."""
    folds = generate_walk_forward_folds(
        timestamps=overall[
            "timestamp"
        ]
    )

    fold_metrics: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for fold in folds:

        train_df = overall[
            overall["timestamp"]
            <= fold.train_end
        ].copy()

        validation_df = overall[
            (
                overall["timestamp"]
                >= fold.validation_start
            )
            & (
                overall["timestamp"]
                <= fold.validation_end
            )
        ].copy()

        result = evaluate_exponential_smoothing_fold(
            fold_number=fold.fold_number,
            train_df=train_df,
            validation_df=validation_df,
        )

        metrics = result["metrics"]

        fold_metrics.append(
            {
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
            validation_df["timestamp"]
        ):
            prediction_rows.append(
                {
                    "fold":
                        fold.fold_number,
                    "timestamp":
                        timestamp,
                    "actual":
                        result["actual"][index],
                    "predicted":
                        result["predicted"][index],
                    "lower_80":
                        result["lower"][index],
                    "upper_80":
                        result["upper"][index],
                }
            )

        print(
            f"Exponential Smoothing fold "
            f"{fold.fold_number}/{len(folds)} | "
            f"MAPE={metrics['mape']:.2f}% | "
            f"RMSE={metrics['rmse']:.2f} | "
            f"MAE={metrics['mae']:.2f}"
        )

    return (
        pd.DataFrame(fold_metrics),
        pd.DataFrame(prediction_rows),
    )

def evaluate_prophet_fold(
    fold_number: int,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> dict[str, object]:
    """
    Train Prophet on one expanding-window training fold and
    generate the full 4-week validation forecast natively.

    Prophet does not use the lag-based recursive mechanism used
    by XGBoost, so actual validation targets are never fed back
    into the forecast.
    """

    prophet_train = pd.DataFrame(
        {
            "ds": train_df["timestamp"],
            "y": train_df[TARGET_COLUMN],
        }
    )

    model = Prophet(
        interval_width=0.80,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
    )

    model.fit(
        prophet_train
    )

    future = pd.DataFrame(
        {
            "ds": validation_df[
                "timestamp"
            ].to_numpy()
        }
    )

    forecast = model.predict(
        future
    )

    predicted = np.maximum(
        forecast["yhat"].to_numpy(
            dtype=float
        ),
        0.0,
    )

    lower = np.maximum(
        forecast["yhat_lower"].to_numpy(
            dtype=float
        ),
        0.0,
    )

    upper = np.maximum(
        forecast["yhat_upper"].to_numpy(
            dtype=float
        ),
        lower,
    )

    actual = validation_df[
        TARGET_COLUMN
    ].to_numpy(
        dtype=float
    )

    metrics = calculate_fold_metrics(
        actual=actual,
        predicted=predicted,
        lower=lower,
        upper=upper,
    )

    return {
        "fold": fold_number,
        "actual": actual,
        "predicted": predicted,
        "lower": lower,
        "upper": upper,
        "metrics": metrics,
        "model": model,
    }

def walk_forward_prophet(
    overall: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run Prophet across the approved 8 expanding-window folds.
    """

    folds = generate_walk_forward_folds(
        timestamps=overall[
            "timestamp"
        ]
    )

    fold_metrics: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for fold in folds:

        train_df = overall[
            overall["timestamp"]
            <= fold.train_end
        ].copy()

        validation_df = overall[
            (
                overall["timestamp"]
                >= fold.validation_start
            )
            & (
                overall["timestamp"]
                <= fold.validation_end
            )
        ].copy()

        if len(train_df) != (
            52
            + (fold.fold_number - 1) * 4
        ):
            raise ValueError(
                f"Unexpected training size "
                f"for Prophet fold "
                f"{fold.fold_number}: "
                f"{len(train_df)}"
            )

        if len(validation_df) != 4:
            raise ValueError(
                f"Unexpected validation size "
                f"for Prophet fold "
                f"{fold.fold_number}: "
                f"{len(validation_df)}"
            )

        result = evaluate_prophet_fold(
            fold_number=fold.fold_number,
            train_df=train_df,
            validation_df=validation_df,
        )

        metrics = result["metrics"]

        fold_metrics.append(
            {
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
            validation_df["timestamp"]
        ):
            prediction_rows.append(
                {
                    "fold":
                        fold.fold_number,
                    "timestamp":
                        timestamp,
                    "actual":
                        result["actual"][index],
                    "predicted":
                        result["predicted"][index],
                    "lower_80":
                        result["lower"][index],
                    "upper_80":
                        result["upper"][index],
                }
            )

        print(
            f"Prophet fold "
            f"{fold.fold_number}/{len(folds)} | "
            f"MAPE={metrics['mape']:.2f}% | "
            f"RMSE={metrics['rmse']:.2f} | "
            f"MAE={metrics['mae']:.2f}"
        )

    return (
        pd.DataFrame(fold_metrics),
        pd.DataFrame(prediction_rows),
    )


# ============================================================================
# MLflow LOGGING
# ============================================================================

def log_model_experiment(
    model_name: str,
    aggregate_metrics: dict[str, float],
    parameters: dict[str, object],
    fold_metrics: pd.DataFrame,
) -> None:
    """Log one model comparison result to MLflow."""

    with mlflow.start_run(
        run_name=f"overall_{model_name}",
    ):
        mlflow.log_params(
            parameters
        )

        mlflow.log_metrics(
            {
                key: float(value)
                for key, value
                in aggregate_metrics.items()
                if np.isfinite(value)
            }
        )

        fold_path = (
            REPORT_DIR
            / f"mlflow_{model_name}_fold_metrics.csv"
        )

        fold_metrics.to_csv(
            fold_path,
            index=False,
        )

        mlflow.log_artifact(
            str(fold_path)
        )


# ============================================================================
# MODEL COMPARISON
# ============================================================================

def build_model_comparison(
    xgb_metrics: pd.DataFrame,
    ets_metrics: pd.DataFrame,
    prophet_metrics: pd.DataFrame,
    lstm_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Create the overall model comparison table."""

    xgb_aggregate = aggregate_fold_metrics(
        xgb_metrics
    )

    ets_aggregate = aggregate_fold_metrics(
        ets_metrics
    )

    prophet_aggregate = aggregate_fold_metrics(
        prophet_metrics
    )

    lstm_aggregate = aggregate_fold_metrics(
        lstm_metrics
    )

    comparison = pd.DataFrame(
        [
            {
                "model":
                    "xgboost",
                **xgb_aggregate,
            },
            {
                "model":
                    "exponential_smoothing",
                **ets_aggregate,
            },
            {
                "model":
                    "prophet",
                **prophet_aggregate,
            },
            {
                "model":
                    "lstm",
                **lstm_aggregate,
            },
        ]
    )

    comparison = comparison.sort_values(
        "mean_mape",
        ascending=True,
    ).reset_index(
        drop=True
    )

    comparison["rank"] = (
        np.arange(
            len(comparison)
        )
        + 1
    )

    return comparison

# ============================================================================
# WEIGHTED ENSEMBLE
# ============================================================================

def calculate_ensemble_weights(
    comparison: pd.DataFrame,
) -> dict[str, float]:
    """
    Calculate model weights using inverse mean MAPE.

    Lower MAPE receives a larger weight.

    The weights are normalized to sum to 1.
    """

    valid = comparison[
        comparison["mean_mape"].notna()
        & (comparison["mean_mape"] > 0)
    ].copy()

    if valid.empty:
        raise ValueError(
            "No valid model MAPE values available "
            "for ensemble weighting."
        )

    inverse_mape = (
        1.0 / valid["mean_mape"]
    )

    normalized_weights = (
        inverse_mape
        / inverse_mape.sum()
    )

    weights = dict(
        zip(
            valid["model"],
            normalized_weights,
        )
    )

    return weights

def build_walk_forward_ensemble(
    prediction_frames: dict[str, pd.DataFrame],
    fold_metric_frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate the weighted ensemble without using the current fold's
    validation results to determine its own weights.

    Fold 1:
        equal weights because there is no prior validation history.

    Fold N:
        weights are learned only from folds 1 ... N-1.

    Returns
    -------
    ensemble_predictions:
        One row per validation observation.

    ensemble_fold_metrics:
        Metrics calculated separately for each outer fold.
    """

    model_names = list(
        prediction_frames.keys()
    )

    if not model_names:
        raise ValueError(
            "No prediction frames supplied."
        )

    # Make sure all models contain the same 8 folds.
    available_folds = sorted(
        set(
            prediction_frames[
                model_names[0]
            ]["fold"]
            .astype(int)
            .tolist()
        )
    )

    if available_folds != list(
        range(1, 9)
    ):
        raise ValueError(
            f"Expected folds 1-8, found {available_folds}."
        )

    ensemble_prediction_rows: list[dict] = []
    ensemble_fold_metric_rows: list[dict] = []

    # Keep the actual weights used for each fold for transparency.
    fold_weight_rows: list[dict] = []

    for current_fold in available_folds:

        # ------------------------------------------------------------
        # Learn weights ONLY from previous folds.
        # ------------------------------------------------------------

        if current_fold == 1:

            weights = {
                model_name:
                    1.0 / len(model_names)
                for model_name in model_names
            }

        else:

            inverse_scores: dict[str, float] = {}

            for model_name in model_names:

                historical_metrics = (
                    fold_metric_frames[
                        model_name
                    ]
                    .loc[
                        lambda df:
                            df["fold"]
                            < current_fold
                    ]
                )

                historical_mape = (
                    historical_metrics[
                        "mape"
                    ]
                    .mean()
                )

                if (
                    pd.isna(historical_mape)
                    or historical_mape <= 0
                ):
                    inverse_scores[
                        model_name
                    ] = 0.0
                else:
                    inverse_scores[
                        model_name
                    ] = 1.0 / float(
                        historical_mape
                    )

            total_inverse = sum(
                inverse_scores.values()
            )

            if total_inverse <= 0:
                weights = {
                    model_name:
                        1.0 / len(model_names)
                    for model_name in model_names
                }
            else:
                weights = {
                    model_name:
                        score / total_inverse
                    for model_name, score
                    in inverse_scores.items()
                }

        # ------------------------------------------------------------
        # Store fold-specific weights.
        # ------------------------------------------------------------

        fold_weight_row = {
            "fold": current_fold,
        }

        fold_weight_row.update(
            {
                f"weight_{model_name}":
                    float(weight)
                for model_name, weight
                in weights.items()
            }
        )

        fold_weight_rows.append(
            fold_weight_row
        )

        # ------------------------------------------------------------
        # Align current-fold predictions.
        # ------------------------------------------------------------

        current_frames: list[pd.DataFrame] = []

        for model_name in model_names:

            frame = prediction_frames[
                model_name
            ]

            current = frame.loc[
                frame["fold"] == current_fold,
                [
                    "fold",
                    "timestamp",
                    "actual",
                    "predicted",
                    "lower_80",
                    "upper_80",
                ],
            ].copy()

            if len(current) != 4:
                raise ValueError(
                    f"{model_name}, fold "
                    f"{current_fold}: expected 4 "
                    f"validation rows, found {len(current)}."
                )

            current = current.rename(
                columns={
                    "predicted":
                        f"predicted_{model_name}",
                    "lower_80":
                        f"lower_80_{model_name}",
                    "upper_80":
                        f"upper_80_{model_name}",
                }
            )

            current_frames.append(
                current
            )

        merged = current_frames[0]

        for current in current_frames[1:]:
            merged = merged.merge(
                current,
                on=[
                    "fold",
                    "timestamp",
                    "actual",
                ],
                how="inner",
                validate="one_to_one",
            )

        if len(merged) != 4:
            raise ValueError(
                f"Fold {current_fold}: aligned "
                f"ensemble rows should be 4, "
                f"found {len(merged)}."
            )

        # ------------------------------------------------------------
        # Weighted ensemble prediction.
        # ------------------------------------------------------------

        merged["predicted"] = 0.0
        merged["lower_80"] = 0.0
        merged["upper_80"] = 0.0

        for model_name, weight in weights.items():

            merged["predicted"] += (
                weight
                * merged[
                    f"predicted_{model_name}"
                ]
            )

            merged["lower_80"] += (
                weight
                * merged[
                    f"lower_80_{model_name}"
                ]
            )

            merged["upper_80"] += (
                weight
                * merged[
                    f"upper_80_{model_name}"
                ]
            )

        merged["predicted"] = np.maximum(
            merged["predicted"],
            0.0,
        )

        merged["lower_80"] = np.maximum(
            merged["lower_80"],
            0.0,
        )

        merged["upper_80"] = np.maximum(
            merged["upper_80"],
            merged["lower_80"],
        )

        # ------------------------------------------------------------
        # Evaluate THIS fold using weights determined from PRIOR folds.
        # ------------------------------------------------------------

        fold_metrics = calculate_metrics(
            actual=merged[
                "actual"
            ].to_numpy(
                dtype=float
            ),
            predicted=merged[
                "predicted"
            ].to_numpy(
                dtype=float
            ),
            lower=merged[
                "lower_80"
            ].to_numpy(
                dtype=float
            ),
            upper=merged[
                "upper_80"
            ].to_numpy(
                dtype=float
            ),
        )

        ensemble_fold_metric_rows.append(
            {
                "fold":
                    current_fold,
                "validation_start":
                    merged[
                        "timestamp"
                    ].min(),
                "validation_end":
                    merged[
                        "timestamp"
                    ].max(),
                "mae":
                    fold_metrics["mae"],
                "rmse":
                    fold_metrics["rmse"],
                "mape":
                    fold_metrics["mape"],
                "interval_coverage_pct":
                    fold_metrics[
                        "interval_coverage_pct"
                    ],
            }
        )

        # ------------------------------------------------------------
        # Keep prediction-level results.
        # ------------------------------------------------------------

        for _, row in merged.iterrows():

            ensemble_prediction_rows.append(
                {
                    "fold":
                        int(row["fold"]),
                    "timestamp":
                        row["timestamp"],
                    "actual":
                        float(row["actual"]),
                    "predicted":
                        float(row["predicted"]),
                    "lower_80":
                        float(row["lower_80"]),
                    "upper_80":
                        float(row["upper_80"]),
                }
            )

        print(
            f"Ensemble fold "
            f"{current_fold}/8 | "
            f"MAPE={fold_metrics['mape']:.2f}% | "
            f"RMSE={fold_metrics['rmse']:.2f} | "
            f"MAE={fold_metrics['mae']:.2f}"
        )

    ensemble_predictions = pd.DataFrame(
        ensemble_prediction_rows
    )

    ensemble_fold_metrics = pd.DataFrame(
        ensemble_fold_metric_rows
    )

    ensemble_fold_metrics.attrs[
        "fold_weights"
    ] = pd.DataFrame(
        fold_weight_rows
    )

    return (
        ensemble_predictions,
        ensemble_fold_metrics,
    )

def build_ensemble_predictions(
    prediction_frames: dict[str, pd.DataFrame],
    weights: dict[str, float],
) -> pd.DataFrame:
    """
    Combine the fold-aligned out-of-fold predictions from the
    four base models.

    All models must contain the same fold/timestamp observations.
    """

    required_keys = {
        "fold",
        "timestamp",
        "actual",
        "predicted",
        "lower_80",
        "upper_80",
    }

    prepared_frames: list[pd.DataFrame] = []

    for model_name, frame in prediction_frames.items():

        missing = required_keys - set(
            frame.columns
        )

        if missing:
            raise ValueError(
                f"{model_name} prediction frame is missing "
                f"columns: {sorted(missing)}"
            )

        prepared = frame[
            [
                "fold",
                "timestamp",
                "actual",
                "predicted",
                "lower_80",
                "upper_80",
            ]
        ].copy()

        prepared = prepared.rename(
            columns={
                "predicted":
                    f"predicted_{model_name}",
                "lower_80":
                    f"lower_80_{model_name}",
                "upper_80":
                    f"upper_80_{model_name}",
            }
        )

        prepared_frames.append(
            prepared
        )

    if not prepared_frames:
        raise ValueError(
            "No prediction frames were supplied."
        )

    merged = prepared_frames[0]

    for frame in prepared_frames[1:]:
        merged = merged.merge(
            frame,
            on=[
                "fold",
                "timestamp",
                "actual",
            ],
            how="inner",
            validate="one_to_one",
        )

    expected_rows = 8 * 4

    if len(merged) != expected_rows:
        raise ValueError(
            "Expected "
            f"{expected_rows} aligned out-of-fold "
            f"observations, found {len(merged)}."
        )

    merged["predicted"] = 0.0
    merged["lower_80"] = 0.0
    merged["upper_80"] = 0.0

    for model_name, weight in weights.items():

        merged["predicted"] += (
            weight
            * merged[
                f"predicted_{model_name}"
            ]
        )

        merged["lower_80"] += (
            weight
            * merged[
                f"lower_80_{model_name}"
            ]
        )

        merged["upper_80"] += (
            weight
            * merged[
                f"upper_80_{model_name}"
            ]
        )

    merged["predicted"] = np.maximum(
        merged["predicted"],
        0.0,
    )

    merged["lower_80"] = np.maximum(
        merged["lower_80"],
        0.0,
    )

    merged["upper_80"] = np.maximum(
        merged["upper_80"],
        merged["lower_80"],
    )

    return merged[
        [
            "fold",
            "timestamp",
            "actual",
            "predicted",
            "lower_80",
            "upper_80",
        ]
    ].sort_values(
        [
            "fold",
            "timestamp",
        ]
    ).reset_index(
        drop=True
    )


# ============================================================================
# STANDALONE STAGE-1 PIPELINE
# ============================================================================

def run_stage1_training() -> None:
    """
    Run the complete Stage-1 overall-revenue ML pipeline.

    Models:
        1. XGBoost
        2. Exponential Smoothing
        3. Prophet
        4. LSTM
        5. Walk-forward weighted ensemble

    Validation:
        52-week expanding training window
        4-week validation horizon
        4-week step
        8 outer folds

    Important:
        Ensemble weights for each validation fold are learned only
        from earlier folds, preventing the current fold's actual
        outcomes from influencing its own ensemble prediction.
    """
    set_random_seed(42)
    
    print(
        "=== ML STAGE 1 ==="
    )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    ensure_directories()
    configure_mlflow()

    # ------------------------------------------------------------------
    # Load canonical feature dataset
    # ------------------------------------------------------------------

    print(
        "\nLoading canonical feature dataset..."
    )

    features = load_feature_dataset()

    print(
        f"Loaded {len(features):,} rows."
    )

    # ------------------------------------------------------------------
    # Overall series
    # ------------------------------------------------------------------

    overall = get_overall_series(
        features
    )

    print(
        "\n=== OVERALL SERIES ==="
    )

    print(
        f"Rows: {len(overall)}"
    )

    print(
        f"Start: "
        f"{overall['timestamp'].min()}"
    )

    print(
        f"End: "
        f"{overall['timestamp'].max()}"
    )

    print(
        f"Total revenue: "
        f"{overall[TARGET_COLUMN].sum():,.2f}"
    )

    # ==================================================================
    # 1. XGBOOST
    # ==================================================================

    print(
        "\n=== XGBOOST WALK-FORWARD ==="
    )

    xgb_fold_metrics, xgb_predictions = (
        walk_forward_xgb(
            overall
        )
    )

    xgb_aggregate = aggregate_fold_metrics(
        xgb_fold_metrics
    )

    print(
        "\n=== AGGREGATED XGBOOST METRICS ==="
    )

    for key, value in xgb_aggregate.items():
        print(
            f"{key}: {value:.4f}"
        )

    # ==================================================================
    # 2. EXPONENTIAL SMOOTHING
    # ==================================================================

    print(
        "\n=== EXPONENTIAL SMOOTHING WALK-FORWARD ==="
    )

    ets_fold_metrics, ets_predictions = (
        walk_forward_exponential_smoothing(
            overall
        )
    )

    ets_aggregate = aggregate_fold_metrics(
        ets_fold_metrics
    )

    print(
        "\n=== AGGREGATED EXPONENTIAL SMOOTHING METRICS ==="
    )

    for key, value in ets_aggregate.items():
        print(
            f"{key}: {value:.4f}"
        )

    # ==================================================================
    # 3. PROPHET
    # ==================================================================

    print(
        "\n=== PROPHET WALK-FORWARD ==="
    )

    prophet_fold_metrics, prophet_predictions = (
        walk_forward_prophet(
            overall
        )
    )

    prophet_aggregate = aggregate_fold_metrics(
        prophet_fold_metrics
    )

    print(
        "\n=== AGGREGATED PROPHET METRICS ==="
    )

    for key, value in prophet_aggregate.items():
        print(
            f"{key}: {value:.4f}"
        )

    # ==================================================================
    # 4. LSTM
    # ==================================================================

    print(
        "\n=== LSTM WALK-FORWARD ==="
    )

    lstm_fold_metrics, lstm_predictions = (
        walk_forward_lstm(
            overall
        )
    )

    lstm_aggregate = aggregate_fold_metrics(
        lstm_fold_metrics
    )

    print(
        "\n=== AGGREGATED LSTM METRICS ==="
    )

    for key, value in lstm_aggregate.items():
        print(
            f"{key}: {value:.4f}"
        )

    # ==================================================================
    # BASE MODEL COMPARISON
    # ==================================================================

    comparison = build_model_comparison(
        xgb_metrics=xgb_fold_metrics,
        ets_metrics=ets_fold_metrics,
        prophet_metrics=prophet_fold_metrics,
        lstm_metrics=lstm_fold_metrics,
    )

    print(
        "\n=== FOUR-MODEL COMPARISON ==="
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    # ==================================================================
    # PREPARE OUT-OF-FOLD PREDICTIONS
    # ==================================================================

    prediction_frames = {
        "xgboost":
            xgb_predictions,
        "exponential_smoothing":
            ets_predictions,
        "prophet":
            prophet_predictions,
        "lstm":
            lstm_predictions,
    }

    fold_metric_frames = {
        "xgboost":
            xgb_fold_metrics,
        "exponential_smoothing":
            ets_fold_metrics,
        "prophet":
            prophet_fold_metrics,
        "lstm":
            lstm_fold_metrics,
    }

    # ==================================================================
    # 5. LEAKAGE-SAFE WALK-FORWARD ENSEMBLE
    # ==================================================================

    print(
        "\n=== WALK-FORWARD WEIGHTED ENSEMBLE ==="
    )

    (
        ensemble_predictions,
        ensemble_fold_metrics,
    ) = build_walk_forward_ensemble(
        prediction_frames=prediction_frames,
        fold_metric_frames=fold_metric_frames,
    )

    # Aggregate the eight ensemble folds.
    ensemble_aggregate = aggregate_fold_metrics(
        ensemble_fold_metrics
    )

    print(
        "\n=== WALK-FORWARD ENSEMBLE METRICS ==="
    )

    for key, value in ensemble_aggregate.items():
        print(
            f"{key}: {value:.4f}"
        )

    # ==================================================================
    # FINAL FIVE-MODEL COMPARISON
    # ==================================================================

    ensemble_row = {
        "model":
            "weighted_ensemble",
        **ensemble_aggregate,
    }

    comparison = pd.concat(
        [
            comparison,
            pd.DataFrame(
                [ensemble_row]
            ),
        ],
        ignore_index=True,
    )

    comparison = comparison.sort_values(
        "mean_mape",
        ascending=True,
    ).reset_index(
        drop=True
    )

    comparison["rank"] = (
        np.arange(
            len(comparison)
        )
        + 1
    )

    print(
        "\n=== FINAL 5-MODEL COMPARISON ==="
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    best_model = str(
        comparison.iloc[0]["model"]
    )

    print(
        f"\nStage-1 best model: "
        f"{best_model}"
    )

    # ==================================================================
    # SAVE FOLD WEIGHTS
    # ==================================================================

    fold_weights = ensemble_fold_metrics.attrs.get(
        "fold_weights"
    )

    if fold_weights is not None:

        fold_weights_path = (
            REPORT_DIR
            / "stage1_ensemble_fold_weights.csv"
        )

        fold_weights.to_csv(
            fold_weights_path,
            index=False,
        )

    # ==================================================================
    # SAVE ENSEMBLE PREDICTIONS
    # ==================================================================

    ensemble_predictions_path = (
        REPORT_DIR
        / "stage1_weighted_ensemble_predictions.csv"
    )

    ensemble_predictions.to_csv(
        ensemble_predictions_path,
        index=False,
    )

    # ==================================================================
    # SAVE FINAL MODEL COMPARISON
    # ==================================================================

    comparison.to_csv(
        STAGE1_REPORT_PATH,
        index=False,
    )

    # ==================================================================
    # SAVE INDIVIDUAL MODEL FOLD REPORTS
    # ==================================================================

    xgb_fold_metrics.to_csv(
        XGB_FOLD_REPORT_PATH,
        index=False,
    )

    ets_fold_metrics.to_csv(
        ETS_FOLD_REPORT_PATH,
        index=False,
    )

    prophet_fold_metrics.to_csv(
        REPORT_DIR
        / "stage1_prophet_fold_metrics.csv",
        index=False,
    )

    lstm_fold_metrics.to_csv(
        REPORT_DIR
        / "stage1_lstm_fold_metrics.csv",
        index=False,
    )

    # ==================================================================
    # MLFLOW — INDIVIDUAL MODELS
    # ==================================================================

    log_model_experiment(
        model_name="xgboost",
        aggregate_metrics=xgb_aggregate,
        parameters={
            **XGB_PARAMS,
            "series_type":
                "overall",
            "series_id":
                "overall",
            "initial_train_weeks":
                52,
            "validation_horizon":
                4,
            "step_weeks":
                4,
            "recursive_forecasting":
                True,
        },
        fold_metrics=xgb_fold_metrics,
    )

    log_model_experiment(
        model_name="exponential_smoothing",
        aggregate_metrics=ets_aggregate,
        parameters={
            "series_type":
                "overall",
            "series_id":
                "overall",
            "initial_train_weeks":
                52,
            "validation_horizon":
                4,
            "step_weeks":
                4,
            "recursive_forecasting":
                False,
        },
        fold_metrics=ets_fold_metrics,
    )

    log_model_experiment(
        model_name="prophet",
        aggregate_metrics=prophet_aggregate,
        parameters={
            "series_type":
                "overall",
            "series_id":
                "overall",
            "initial_train_weeks":
                52,
            "validation_horizon":
                4,
            "step_weeks":
                4,
            "interval_width":
                0.80,
            "yearly_seasonality":
                True,
            "weekly_seasonality":
                False,
            "daily_seasonality":
                False,
            "forecasting_mode":
                "native_multi_step",
        },
        fold_metrics=prophet_fold_metrics,
    )

    log_model_experiment(
        model_name="lstm",
        aggregate_metrics=lstm_aggregate,
        parameters={
            "series_type":
                "overall",
            "series_id":
                "overall",
            "initial_train_weeks":
                52,
            "validation_horizon":
                4,
            "step_weeks":
                4,
            "recursive_forecasting":
                True,
            "sequence_length":
                LSTM_SEQUENCE_LENGTH,
            "hidden_size":
                LSTM_HIDDEN_SIZE,
            "epochs":
                LSTM_EPOCHS,
            "learning_rate":
                LSTM_LEARNING_RATE,
        },
        fold_metrics=lstm_fold_metrics,
    )

    # ==================================================================
    # MLFLOW — LEAKAGE-SAFE ENSEMBLE
    # ==================================================================

    fold_weights_path = (
        REPORT_DIR
        / "stage1_ensemble_fold_weights.csv"
    )

    ensemble_predictions_path = (
        REPORT_DIR
        / "stage1_weighted_ensemble_predictions.csv"
    )

    with mlflow.start_run(
        run_name="overall_weighted_ensemble_walk_forward"
    ):

        mlflow.log_params(
            {
                "series_type":
                    "overall",
                "series_id":
                    "overall",
                "weighting_method":
                    "inverse_previous_fold_mape",
                "initial_train_weeks":
                    52,
                "validation_horizon":
                    4,
                "step_weeks":
                    4,
                "recursive_xgboost":
                    True,
                "recursive_lstm":
                    True,
                "ensemble_evaluation":
                    "past_fold_weights_only",
            }
        )

        mlflow.log_metrics(
            {
                "mean_mae":
                    float(
                        ensemble_aggregate[
                            "mean_mae"
                        ]
                    ),
                "mean_rmse":
                    float(
                        ensemble_aggregate[
                            "mean_rmse"
                        ]
                    ),
                "mean_mape":
                    float(
                        ensemble_aggregate[
                            "mean_mape"
                        ]
                    ),
                "mean_interval_coverage_pct":
                    float(
                        ensemble_aggregate[
                            "mean_interval_coverage_pct"
                        ]
                    ),
            }
        )

        if fold_weights_path.is_file():
            mlflow.log_artifact(
                str(fold_weights_path)
            )

        if ensemble_predictions_path.is_file():
            mlflow.log_artifact(
                str(ensemble_predictions_path)
            )

    # ==================================================================
    # FINAL OUTPUT SUMMARY
    # ==================================================================

    print(
        "\n=== STAGE 1 COMPLETE ==="
    )

    print(
        f"Final model comparison saved to:\n"
        f"{STAGE1_REPORT_PATH}"
    )

    print(
        "\nEnsemble fold weights saved to:"
    )

    print(
        REPORT_DIR
        / "stage1_ensemble_fold_weights.csv"
    )

    print(
        "\nEnsemble predictions saved to:"
    )

    print(
        REPORT_DIR
        / "stage1_weighted_ensemble_predictions.csv"
    )

    print(
        "\nMLflow tracking URI:"
    )

    print(
        mlflow.get_tracking_uri()
    )

if __name__ == "__main__":
    run_stage1_training()