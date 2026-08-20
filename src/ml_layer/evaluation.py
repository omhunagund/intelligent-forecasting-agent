"""
Time-series evaluation utilities for the Intelligent Business Forecasting Agent.

Approved validation design
--------------------------
Initial training window : 52 weeks
Validation horizon       : 4 weeks
Step size                : 4 weeks
Window type              : expanding

The functions in this module are model-agnostic. They do not train
models and do not decide which forecasting algorithm is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


# ============================================================================
# APPROVED WALK-FORWARD SETTINGS
# ============================================================================

INITIAL_TRAIN_WEEKS = 52
VALIDATION_HORIZON = 4
STEP_WEEKS = 4


# ============================================================================
# FOLD DESCRIPTION
# ============================================================================

@dataclass(frozen=True)
class WalkForwardFold:
    """Description of one expanding-window validation fold."""

    fold_number: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


# ============================================================================
# WALK-FORWARD SPLITS
# ============================================================================

def generate_walk_forward_folds(
    timestamps: pd.Series | pd.DatetimeIndex,
    initial_train_weeks: int = INITIAL_TRAIN_WEEKS,
    validation_horizon: int = VALIDATION_HORIZON,
    step_weeks: int = STEP_WEEKS,
) -> list[WalkForwardFold]:
    """
    Generate expanding-window weekly validation folds.

    Example
    -------
    Fold 1:
        train weeks 1-52
        validate weeks 53-56

    Fold 2:
        train weeks 1-56
        validate weeks 57-60
    """
    dates = pd.DatetimeIndex(
        pd.to_datetime(timestamps)
    ).sort_values().unique()

    if len(dates) < initial_train_weeks + validation_horizon:
        raise ValueError(
            "Not enough weekly observations for the requested "
            "walk-forward configuration."
        )

    if validation_horizon <= 0:
        raise ValueError(
            "validation_horizon must be positive."
        )

    if step_weeks <= 0:
        raise ValueError(
            "step_weeks must be positive."
        )

    folds: list[WalkForwardFold] = []

    train_end_index = initial_train_weeks

    fold_number = 1

    while (
        train_end_index + validation_horizon
        <= len(dates)
    ):
        validation_start_index = train_end_index
        validation_end_index = (
            train_end_index + validation_horizon
        )

        folds.append(
            WalkForwardFold(
                fold_number=fold_number,
                train_start=dates[0],
                train_end=dates[train_end_index - 1],
                validation_start=dates[
                    validation_start_index
                ],
                validation_end=dates[
                    validation_end_index - 1
                ],
            )
        )

        train_end_index += step_weeks
        fold_number += 1

    return folds


# ============================================================================
# ERROR METRICS
# ============================================================================

def mae(
    actual: Iterable[float],
    predicted: Iterable[float],
) -> float:
    """Mean Absolute Error."""
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)

    return float(
        np.mean(
            np.abs(y_true - y_pred)
        )
    )


def rmse(
    actual: Iterable[float],
    predicted: Iterable[float],
) -> float:
    """Root Mean Squared Error."""
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)

    return float(
        np.sqrt(
            np.mean(
                (y_true - y_pred) ** 2
            )
        )
    )


def mape(
    actual: Iterable[float],
    predicted: Iterable[float],
    epsilon: float = 1e-8,
) -> float:
    """
    Mean Absolute Percentage Error.

    Zero actual values are ignored rather than causing division
    by zero.
    """
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)

    denominator_mask = np.abs(y_true) > epsilon

    if not denominator_mask.any():
        return float("nan")

    return float(
        np.mean(
            np.abs(
                (
                    y_true[denominator_mask]
                    - y_pred[denominator_mask]
                )
                / y_true[denominator_mask]
            )
        )
        * 100
    )


def interval_coverage(
    actual: Iterable[float],
    lower: Iterable[float],
    upper: Iterable[float],
) -> float:
    """
    Empirical prediction-interval coverage.

    Returns percentage of actual observations lying inside
    [lower, upper].
    """
    y_true = np.asarray(actual, dtype=float)
    y_lower = np.asarray(lower, dtype=float)
    y_upper = np.asarray(upper, dtype=float)

    covered = (
        (y_true >= y_lower)
        & (y_true <= y_upper)
    )

    return float(
        covered.mean() * 100
    )


# ============================================================================
# METRIC SUMMARY
# ============================================================================

def calculate_metrics(
    actual: Iterable[float],
    predicted: Iterable[float],
    lower: Iterable[float] | None = None,
    upper: Iterable[float] | None = None,
) -> dict[str, float]:
    """
    Calculate the project's approved evaluation metrics.
    """
    metrics = {
        "mae": mae(
            actual,
            predicted,
        ),
        "rmse": rmse(
            actual,
            predicted,
        ),
        "mape": mape(
            actual,
            predicted,
        ),
    }

    if (
        lower is not None
        and upper is not None
    ):
        metrics["interval_coverage_pct"] = (
            interval_coverage(
                actual,
                lower,
                upper,
            )
        )

    return metrics


# ============================================================================
# FOLD RESULT COLLECTION
# ============================================================================

def evaluate_fold(
    fold_number: int,
    timestamps: Iterable[pd.Timestamp],
    actual: Iterable[float],
    predicted: Iterable[float],
    lower: Iterable[float] | None = None,
    upper: Iterable[float] | None = None,
) -> dict[str, float | int | str]:
    """
    Evaluate one walk-forward validation fold.
    """
    timestamp_index = pd.DatetimeIndex(
        pd.to_datetime(timestamps)
    )

    metrics = calculate_metrics(
        actual,
        predicted,
        lower,
        upper,
    )

    result: dict[str, float | int | str] = {
        "fold": fold_number,
        "validation_start": timestamp_index.min().isoformat(),
        "validation_end": timestamp_index.max().isoformat(),
        **metrics,
    }

    return result


def aggregate_fold_metrics(
    fold_results: pd.DataFrame,
) -> dict[str, float]:
    """
    Average metrics across validation folds.

    RMSE is averaged across folds rather than reconstructed
    from pooled predictions so each fold contributes equally.
    """
    metric_columns = [
        "mae",
        "rmse",
        "mape",
        "interval_coverage_pct",
    ]

    available = [
        column
        for column in metric_columns
        if column in fold_results.columns
    ]

    return {
        f"mean_{column}":
            float(
                fold_results[column].mean(
                    skipna=True
                )
            )
        for column in available
    }


# ============================================================================
# MODEL COMPARISON
# ============================================================================

def rank_models(
    model_metrics: pd.DataFrame,
    primary_metric: str = "mape",
) -> pd.DataFrame:
    """
    Rank models by the requested metric.

    Lower is better for MAPE, MAE, and RMSE.

    Prediction-interval coverage is not used as a primary
    minimization metric.
    """
    if primary_metric not in model_metrics.columns:
        raise KeyError(
            f"Metric '{primary_metric}' not found."
        )

    ranked = model_metrics.copy()

    ranked = ranked.sort_values(
        primary_metric,
        ascending=True,
        na_position="last",
    ).reset_index(drop=True)

    ranked["rank"] = (
        np.arange(len(ranked))
        + 1
    )

    return ranked


# ============================================================================
# WALK-FORWARD VALIDATION HELPER
# ============================================================================

def validate_fold_configuration(
    timestamps: Iterable[pd.Timestamp],
    initial_train_weeks: int = INITIAL_TRAIN_WEEKS,
    validation_horizon: int = VALIDATION_HORIZON,
    step_weeks: int = STEP_WEEKS,
) -> dict[str, int]:
    """
    Validate the approved walk-forward setup and return fold counts.
    """
    folds = generate_walk_forward_folds(
        timestamps=timestamps,
        initial_train_weeks=initial_train_weeks,
        validation_horizon=validation_horizon,
        step_weeks=step_weeks,
    )

    return {
        "total_observations": len(
            pd.DatetimeIndex(
                pd.to_datetime(timestamps)
            ).unique()
        ),
        "fold_count": len(folds),
        "initial_train_weeks": initial_train_weeks,
        "validation_horizon": validation_horizon,
        "step_weeks": step_weeks,
    }


# ============================================================================
# SMOKE TEST
# ============================================================================

def main() -> None:
    """Validate the approved walk-forward configuration."""
    print("=== EVALUATION FRAMEWORK TEST ===")

    timestamps = pd.date_range(
        start="2017-01-01",
        periods=87,
        freq="W-SUN",
    )

    configuration = validate_fold_configuration(
        timestamps
    )

    print("\n=== VALIDATION CONFIGURATION ===")

    for key, value in configuration.items():
        print(
            f"{key}: {value}"
        )

    folds = generate_walk_forward_folds(
        timestamps
    )

    print(
        f"\nGenerated folds: {len(folds)}"
    )

    print("\n=== FOLD WINDOWS ===")

    for fold in folds:
        print(
            f"Fold {fold.fold_number}: "
            f"train {fold.train_start.date()} → "
            f"{fold.train_end.date()} | "
            f"validate {fold.validation_start.date()} → "
            f"{fold.validation_end.date()}"
        )

    # Simple metric smoke test.
    actual = np.array(
        [100.0, 110.0, 120.0, 130.0]
    )

    predicted = np.array(
        [95.0, 112.0, 118.0, 128.0]
    )

    lower = predicted - 10
    upper = predicted + 10

    print("\n=== METRIC TEST ===")

    metrics = calculate_metrics(
        actual,
        predicted,
        lower,
        upper,
    )

    for key, value in metrics.items():
        print(
            f"{key}: {value:.4f}"
        )


if __name__ == "__main__":
    main()