"""
Feature engineering for the Intelligent Business Forecasting Agent.

Responsibilities
----------------
1. Build a long-format weekly forecasting dataset containing:
   - 1 overall revenue series
   - 15 business-category series
   - 5 regional series

2. Generate reusable time-series features per series:
   - lag_1
   - lag_4
   - lag_52
   - rolling_mean_4
   - rolling_std_4
   - month
   - quarter
   - year
   - week_of_year
   - days_to_month_end
   - month_sin
   - month_cos

3. Preserve the original target value as `target_revenue`.

Important
---------
- Raw source data is never modified.
- Feature engineering is performed on weekly aggregated data.
- Features are calculated separately within each series_id.
- Missing feature values during the warm-up period are preserved.
  They are not silently dropped here because the later ML layer
  is responsible for deciding how each model handles warm-up rows.
- Promotional flags, regional population, holiday calendars, and
  other external domain variables are NOT fabricated here because
  they are not present in the approved Olist source data.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

from src.data_layer.cleaning import (
    MODEL_END,
    MODEL_START,
    clean_data,
)
from src.data_layer.ingestion import load_raw_data


# ============================================================================
# CONSTANTS
# ============================================================================

WEEKLY_FREQUENCY: Final[str] = "W-SUN"

EXPECTED_CATEGORY_SERIES: Final[int] = 15
EXPECTED_REGION_SERIES: Final[int] = 5


# ============================================================================
# WEEKLY SERIES BUILDING
# ============================================================================

def _weekly_sum(
    dataframe: pd.DataFrame,
    timestamp_column: str,
    value_column: str,
) -> pd.Series:
    """
    Aggregate a value column to weekly Sunday-ending periods.
    """
    series = (
        dataframe
        .set_index(timestamp_column)[value_column]
        .resample(WEEKLY_FREQUENCY)
        .sum()
    )

    return series


def build_overall_series(
    revenue_items: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the overall weekly revenue series.
    """
    weekly = _weekly_sum(
        revenue_items,
        "order_purchase_timestamp",
        "item_revenue",
    )

    result = weekly.rename("target_revenue").reset_index()

    result["series_type"] = "overall"
    result["series_id"] = "overall"

    return result[
        [
            "series_type",
            "series_id",
            "order_purchase_timestamp",
            "target_revenue",
        ]
    ].rename(
        columns={
            "order_purchase_timestamp": "timestamp",
        }
    )


def build_category_series(
    category_items: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build weekly revenue series for the 15 approved
    business categories.
    """
    grouped = (
        category_items
        .set_index("order_purchase_timestamp")
        .groupby("business_category")["item_revenue"]
        .resample(WEEKLY_FREQUENCY)
        .sum()
        .rename("target_revenue")
        .reset_index()
    )

    grouped["series_type"] = "category"
    grouped["series_id"] = grouped["business_category"]

    return grouped[
        [
            "series_type",
            "series_id",
            "order_purchase_timestamp",
            "target_revenue",
        ]
    ].rename(
        columns={
            "order_purchase_timestamp": "timestamp",
        }
    )


def build_region_series(
    region_items: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build weekly revenue series for the 5 approved macro-regions.
    """
    grouped = (
        region_items
        .set_index("order_purchase_timestamp")
        .groupby("region")["item_revenue"]
        .resample(WEEKLY_FREQUENCY)
        .sum()
        .rename("target_revenue")
        .reset_index()
    )

    grouped["series_type"] = "region"
    grouped["series_id"] = grouped["region"]

    return grouped[
        [
            "series_type",
            "series_id",
            "order_purchase_timestamp",
            "target_revenue",
        ]
    ].rename(
        columns={
            "order_purchase_timestamp": "timestamp",
        }
    )


# ============================================================================
# COMPLETE WEEKLY SERIES CALENDAR
# ============================================================================

def _complete_series_calendar(
    series_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensure every series has a complete weekly calendar.

    Missing weeks are represented by zero revenue.

    This is important for lag and rolling calculations because
    the temporal spacing must remain continuous.
    """
    series_dataframe = series_dataframe.copy()

    series_dataframe["timestamp"] = pd.to_datetime(
        series_dataframe["timestamp"]
    )

    series_dataframe = (
        series_dataframe
        .sort_values(
            ["series_type", "series_id", "timestamp"]
        )
        .drop_duplicates(
            subset=[
                "series_type",
                "series_id",
                "timestamp",
            ],
            keep="last",
        )
    )

    completed_groups: list[pd.DataFrame] = []

    for (series_type, series_id), group in series_dataframe.groupby(
        ["series_type", "series_id"],
        sort=True,
    ):
        group = group.set_index("timestamp")

        full_calendar = pd.date_range(
            start=group.index.min(),
            end=group.index.max(),
            freq=WEEKLY_FREQUENCY,
        )

        group = (
            group.reindex(full_calendar)
            .rename_axis("timestamp")
            .reset_index()
        )

        group["series_type"] = series_type
        group["series_id"] = series_id
        group["target_revenue"] = group[
            "target_revenue"
        ].fillna(0.0)

        completed_groups.append(group)

    return pd.concat(
        completed_groups,
        ignore_index=True,
    ).sort_values(
        ["series_type", "series_id", "timestamp"]
    ).reset_index(drop=True)


# ============================================================================
# FEATURE GENERATION
# ============================================================================

def add_time_series_features(
    series_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add lag, rolling, calendar, and cyclic time-series features.

    All lag and rolling calculations are performed independently
    for each series_id to prevent cross-series contamination.
    """
    dataframe = series_dataframe.copy()

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"]
    )

    dataframe = dataframe.sort_values(
        ["series_type", "series_id", "timestamp"]
    ).reset_index(drop=True)

    grouped_target = dataframe.groupby(
        ["series_type", "series_id"],
        sort=False,
    )["target_revenue"]

    # ------------------------------------------------------------------
    # Lag features
    # ------------------------------------------------------------------

    dataframe["lag_1"] = grouped_target.shift(1)
    dataframe["lag_4"] = grouped_target.shift(4)
    dataframe["lag_52"] = grouped_target.shift(52)

    # ------------------------------------------------------------------
    # Rolling statistics
    #
    # Shift first so the current week's target is never included in
    # the rolling calculation. This prevents target leakage.
    # ------------------------------------------------------------------

    shifted_target = grouped_target.shift(1)

    dataframe["rolling_mean_4"] = (
        shifted_target
        .groupby(
            [
                dataframe["series_type"],
                dataframe["series_id"],
            ]
        )
        .transform(
            lambda values: values.rolling(
                window=4,
                min_periods=1,
            ).mean()
        )
    )

    dataframe["rolling_std_4"] = (
        shifted_target
        .groupby(
            [
                dataframe["series_type"],
                dataframe["series_id"],
            ]
        )
        .transform(
            lambda values: values.rolling(
                window=4,
                min_periods=2,
            ).std()
        )
    )

    # ------------------------------------------------------------------
    # Calendar features
    # ------------------------------------------------------------------

    dataframe["year"] = dataframe["timestamp"].dt.year
    dataframe["month"] = dataframe["timestamp"].dt.month
    dataframe["quarter"] = dataframe["timestamp"].dt.quarter

    dataframe["week_of_year"] = (
        dataframe["timestamp"]
        .dt.isocalendar()
        .week
        .astype("int64")
    )

    dataframe["day_of_week"] = (
        dataframe["timestamp"].dt.dayofweek
    )

    dataframe["is_weekend"] = (
        dataframe["day_of_week"] >= 5
    ).astype("int8")

    dataframe["days_to_month_end"] = (
        dataframe["timestamp"].dt.days_in_month
        - dataframe["timestamp"].dt.day
    )

    # ------------------------------------------------------------------
    # Cyclic calendar encodings
    # ------------------------------------------------------------------

    dataframe["month_sin"] = (
        __import__("numpy").sin(
            2
            * __import__("numpy").pi
            * dataframe["month"]
            / 12
        )
    )

    dataframe["month_cos"] = (
        __import__("numpy").cos(
            2
            * __import__("numpy").pi
            * dataframe["month"]
            / 12
        )
    )

    dataframe["week_sin"] = (
        __import__("numpy").sin(
            2
            * __import__("numpy").pi
            * dataframe["week_of_year"]
            / 52
        )
    )

    dataframe["week_cos"] = (
        __import__("numpy").cos(
            2
            * __import__("numpy").pi
            * dataframe["week_of_year"]
            / 52
        )
    )

    return dataframe


# ============================================================================
# MAIN FEATURE-ENGINEERING PIPELINE
# ============================================================================
def build_recursive_feature_row(
    history: pd.Series,
    timestamp: pd.Timestamp,
) -> pd.DataFrame:
    """
    Build one future feature row using only target history available
    up to the immediately preceding timestamp.

    This is the canonical recursive feature builder used during
    multi-step XGBoost forecasting.

    No future actual target values are used.
    """

    import numpy as np

    history = pd.Series(
        history,
        dtype="float64",
    ).dropna()

    timestamp = pd.Timestamp(timestamp)

    def lag_value(lag: int) -> float:
        if len(history) < lag:
            return float("nan")

        return float(history.iloc[-lag])

    lag_1 = lag_value(1)
    lag_4 = lag_value(4)
    lag_52 = lag_value(52)

    recent_4 = history.iloc[-4:]

    rolling_mean_4 = (
        float(recent_4.mean())
        if len(recent_4) >= 1
        else float("nan")
    )

    rolling_std_4 = (
        float(recent_4.std())
        if len(recent_4) >= 2
        else float("nan")
    )

    month = timestamp.month
    week_of_year = int(
        timestamp.isocalendar().week
    )

    day_of_week = timestamp.dayofweek

    row = {
        "lag_1": lag_1,
        "lag_4": lag_4,
        "lag_52": lag_52,
        "rolling_mean_4": rolling_mean_4,
        "rolling_std_4": rolling_std_4,
        "year": timestamp.year,
        "month": month,
        "quarter": timestamp.quarter,
        "week_of_year": week_of_year,
        "day_of_week": day_of_week,
        "is_weekend": int(day_of_week >= 5),
        "days_to_month_end": (
            timestamp.days_in_month
            - timestamp.day
        ),
        "month_sin": np.sin(
            2 * np.pi * month / 12
        ),
        "month_cos": np.cos(
            2 * np.pi * month / 12
        ),
        "week_sin": np.sin(
            2 * np.pi * week_of_year / 52
        ),
        "week_cos": np.cos(
            2 * np.pi * week_of_year / 52
        ),
    }

    return pd.DataFrame([row])


def build_feature_dataset(
    cleaned: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build the complete long-format forecasting feature dataset.

    Returns
    -------
    pd.DataFrame
        One row per series/week containing the target and features.
    """
    required_cleaned = {
        "revenue_items",
        "category_items",
        "region_items",
    }

    missing = required_cleaned - set(cleaned.keys())

    if missing:
        raise KeyError(
            "Missing cleaned datasets: "
            + ", ".join(sorted(missing))
        )

    revenue_items = cleaned["revenue_items"]
    category_items = cleaned["category_items"]
    region_items = cleaned["region_items"]

    overall = build_overall_series(
        revenue_items
    )

    category = build_category_series(
        category_items
    )

    region = build_region_series(
        region_items
    )

    # Combine all three series families.
    combined = pd.concat(
        [
            overall,
            category,
            region,
        ],
        ignore_index=True,
    )

    # Complete weekly calendars before generating lags.
    combined = _complete_series_calendar(
        combined
    )

    # Generate features independently per series.
    features = add_time_series_features(
        combined
    )

    # Restrict the final modeling-ready table to the approved
    # modeling window.
    features = features[
        (features["timestamp"] >= MODEL_START)
        & (features["timestamp"] <= MODEL_END)
    ].copy()

    return features.sort_values(
        [
            "series_type",
            "series_id",
            "timestamp",
        ]
    ).reset_index(drop=True)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_feature_dataset(
    features: pd.DataFrame,
) -> dict[str, object]:
    """
    Validate the structure of the long-format feature dataset.
    """
    expected_columns = {
        "series_type",
        "series_id",
        "timestamp",
        "target_revenue",
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
    }

    lag_52_missing = int(
        features["lag_52"].isna().sum()
    )

    lag_52_missing_pct = (
        lag_52_missing
        / len(features)
        * 100
    )

    lag_1_missing = int(
        features["lag_1"].isna().sum()
    )

    lag_4_missing = int(
        features["lag_4"].isna().sum()
    )

    missing_columns = expected_columns - set(
        features.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing engineered feature columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in sorted(missing_columns)
            )
        )

    if features["target_revenue"].isna().any():
        raise ValueError(
            "Target revenue contains missing values."
        )

    if (features["target_revenue"] < 0).any():
        raise ValueError(
            "Target revenue contains negative values."
        )

    series_counts = (
        features
        .groupby("series_type")["series_id"]
        .nunique()
        .to_dict()
    )

    expected_series_counts = {
        "overall": 1,
        "category": EXPECTED_CATEGORY_SERIES,
        "region": EXPECTED_REGION_SERIES,
    }

    for series_type, expected_count in (
        expected_series_counts.items()
    ):
        actual_count = series_counts.get(
            series_type,
            0,
        )

        if actual_count != expected_count:
            raise ValueError(
                f"Expected {expected_count} {series_type} "
                f"series, found {actual_count}."
            )

    total_series = (
        features[
            ["series_type", "series_id"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    if total_series != 21:
        raise ValueError(
            f"Expected 21 total series, found {total_series}."
        )

    return {
        "rows": len(features),
        "series_count": total_series,
        "overall_series": series_counts.get(
            "overall",
            0,
        ),
        "category_series": series_counts.get(
            "category",
            0,
        ),
        "region_series": series_counts.get(
            "region",
            0,
        ),
        "model_start": features["timestamp"].min(),
        "model_end": features["timestamp"].max(),

        "lag_1_missing_rows": lag_1_missing,
        "lag_4_missing_rows": lag_4_missing,
        "lag_52_missing_rows": lag_52_missing,
        "lag_52_missing_pct": lag_52_missing_pct,
        "lag_52_missing_is_expected": True,
    }


# ============================================================================
# SUMMARY
# ============================================================================

def summarize_features(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return one summary row per forecasting series.
    """
    summary = (
        features
        .groupby(
            [
                "series_type",
                "series_id",
            ]
        )
        .agg(
            periods=("timestamp", "size"),
            total_revenue=("target_revenue", "sum"),
            mean_weekly_revenue=("target_revenue", "mean"),
            zero_weeks=(
                "target_revenue",
                lambda values: int(
                    (values == 0).sum()
                ),
            ),
        )
        .reset_index()
    )

    return summary


# ============================================================================
# STANDALONE SMOKE TEST
# ============================================================================
def validate_target_reconciliation(
    features: pd.DataFrame,
    cleaned: dict[str, pd.DataFrame],
) -> dict[str, float]:
    """
    Verify that engineered target revenue reconciles exactly with
    the cleaned source datasets inside the approved weekly modeling
    window.

    MODEL_END is inclusive at the date level, so the entire final
    day (2018-08-26) must be included.
    """

    revenue_items = cleaned["revenue_items"]
    category_items = cleaned["category_items"]
    region_items = cleaned["region_items"]

    # ------------------------------------------------------------
    # Inclusive weekly modeling window
    # ------------------------------------------------------------
    model_end_exclusive = MODEL_END + pd.Timedelta(days=1)

    # ------------------------------------------------------------
    # Overall expected revenue
    # ------------------------------------------------------------
    overall_expected = (
        revenue_items[
            (revenue_items["order_purchase_timestamp"] >= MODEL_START)
            & (
                revenue_items["order_purchase_timestamp"]
                < model_end_exclusive
            )
        ]["item_revenue"]
        .sum()
    )

    overall_actual = features.loc[
        features["series_type"] == "overall",
        "target_revenue",
    ].sum()

    overall_difference = float(
        overall_actual - overall_expected
    )

    # ------------------------------------------------------------
    # Category expected revenue
    # ------------------------------------------------------------
    category_expected = (
        category_items[
            (category_items["order_purchase_timestamp"] >= MODEL_START)
            & (
                category_items["order_purchase_timestamp"]
                < model_end_exclusive
            )
        ]["item_revenue"]
        .sum()
    )

    category_actual = features.loc[
        features["series_type"] == "category",
        "target_revenue",
    ].sum()

    category_difference = float(
        category_actual - category_expected
    )

    # ------------------------------------------------------------
    # Region expected revenue
    # ------------------------------------------------------------
    region_expected = (
        region_items[
            (region_items["order_purchase_timestamp"] >= MODEL_START)
            & (
                region_items["order_purchase_timestamp"]
                < model_end_exclusive
            )
        ]["item_revenue"]
        .sum()
    )

    region_actual = features.loc[
        features["series_type"] == "region",
        "target_revenue",
    ].sum()

    region_difference = float(
        region_actual - region_expected
    )

    differences = {
        "overall_difference": overall_difference,
        "category_difference": category_difference,
        "region_difference": region_difference,
    }

    tolerance = 1e-6

    if any(
        abs(value) > tolerance
        for value in differences.values()
    ):
        raise ValueError(
            "Target revenue reconciliation failed:\n"
            + "\n".join(
                f"  {key}: {value}"
                for key, value in differences.items()
            )
        )

    return differences


def main() -> None:
    """
    Standalone smoke test for the feature-engineering layer.

    No feature file is written.
    """
    print("=== FEATURE ENGINEERING TEST ===")

    tables = load_raw_data()

    cleaned = clean_data(
        tables
    )

    features = build_feature_dataset(
        cleaned
    )

    validation = validate_feature_dataset(
        features
    )

    reconciliation = validate_target_reconciliation(
        features,
        cleaned,
    )

    print("\n=== TARGET RECONCILIATION ===")

    for key, value in reconciliation.items():
        print(f"{key}: {value:.6f}")

    print("\nFeature engineering completed successfully.")

    print("\n=== FEATURE DATASET VALIDATION ===")

    for key, value in validation.items():
        print(f"{key}: {value}")

    print("\n=== SERIES COUNTS ===")

    series_counts = (
        features
        .groupby("series_type")["series_id"]
        .nunique()
    )

    print(
        series_counts.to_string()
    )

    print("\n=== FEATURE COLUMNS ===")
    for column in features.columns:
        print(f"- {column}")

    print("\n=== SAMPLE ROWS ===")
    print(
        features.head(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()