"""
Data Layer orchestration for the Intelligent Business Forecasting Agent.

Pipeline
--------
1. Load raw Olist data.
2. Clean and validate the data.
3. Persist one item-level processed source of truth.
4. Build the long-format forecasting feature dataset.
5. Persist the feature dataset.
6. Generate an automated data-quality report.

Raw data under data/raw/ is never modified.

Individual responsibilities remain separated:

    ingestion.py
        Raw data loading

    cleaning.py
        Business-rule cleaning and validation

    feature_engineering.py
        Weekly series and forecasting features

    pipeline.py
        Orchestration and persistence
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import pandas as pd

from src.data_layer.cleaning import (
    MODEL_END,
    MODEL_START,
    build_category_items,
    build_region_items,
    clean_data,
    validate_cleaned_data,
)
from src.data_layer.feature_engineering import (
    build_feature_dataset,
    validate_feature_dataset,
    validate_target_reconciliation,
)
from src.data_layer.ingestion import (
    get_project_root,
    load_raw_data,
)


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT: Final[Path] = get_project_root()

DATA_DIR: Final[Path] = PROJECT_ROOT / "data"

PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
FEATURES_DIR: Final[Path] = DATA_DIR / "features"

REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
DATA_QUALITY_DIR: Final[Path] = REPORTS_DIR / "data_quality"


PROCESSED_DATASET_PATH: Final[Path] = (
    PROCESSED_DIR / "processed_dataset.parquet"
)

FEATURE_DATASET_PATH: Final[Path] = (
    FEATURES_DIR / "forecasting_features.parquet"
)

DATA_QUALITY_REPORT_PATH: Final[Path] = (
    DATA_QUALITY_DIR / "data_quality_report.csv"
)


# ============================================================================
# DIRECTORY SETUP
# ============================================================================

def ensure_output_directories() -> None:
    """Create required output directories when they do not exist."""
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FEATURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DATA_QUALITY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================================
# PROCESSED DATASET
# ============================================================================

def build_processed_dataset(
    cleaned: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build the single item-level processed source of truth.

    The processed dataset contains the complete approved revenue
    universe. Category and region assignment remain available at
    item level through enrichment columns needed by downstream
    analysis, but no weekly aggregation is performed here.
    """
    revenue_items = cleaned["revenue_items"].copy()

    # Attach product category at item level.
    processed = revenue_items.copy()

    return processed


def save_processed_dataset(
    processed: pd.DataFrame,
) -> None:
    """
    Persist the item-level processed dataset as Parquet.
    """
    processed.to_parquet(
        PROCESSED_DATASET_PATH,
        index=False,
    )


# ============================================================================
# FEATURE DATASET
# ============================================================================

def save_feature_dataset(
    features: pd.DataFrame,
) -> None:
    """Persist the long-format forecasting feature dataset."""
    features.to_parquet(
        FEATURE_DATASET_PATH,
        index=False,
    )


# ============================================================================
# DATA-QUALITY REPORT
# ============================================================================

def build_quality_report(
    raw_tables: dict[str, pd.DataFrame],
    cleaned: dict[str, pd.DataFrame],
    features: pd.DataFrame,
    cleaning_metrics: dict[str, object],
    feature_metrics: dict[str, object],
    reconciliation: dict[str, float],
) -> pd.DataFrame:
    """
    Build a compact machine-readable data-quality report.

    Each row represents one validation metric.
    """
    rows: list[dict[str, object]] = []

    generated_at = datetime.now(
        timezone.utc
    ).isoformat()

    def add_metric(
        metric: str,
        value: object,
        status: str = "PASS",
        stage: str = "data_quality",
    ) -> None:
        rows.append(
            {
                "generated_at_utc": generated_at,
                "stage": stage,
                "metric": metric,
                "value": value,
                "status": status,
            }
        )

    # ------------------------------------------------------------------
    # Raw ingestion
    # ------------------------------------------------------------------

    add_metric(
        "raw_table_count",
        len(raw_tables),
        stage="ingestion",
    )

    for name, dataframe in raw_tables.items():
        add_metric(
            f"raw_rows_{name}",
            len(dataframe),
            stage="ingestion",
        )

        add_metric(
            f"raw_columns_{name}",
            len(dataframe.columns),
            stage="ingestion",
        )

        add_metric(
            f"raw_missing_cells_{name}",
            int(
                dataframe.isna().sum().sum()
            ),
            stage="ingestion",
        )

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------

    for metric, value in cleaning_metrics.items():
        add_metric(
            metric,
            value,
            stage="cleaning",
        )

    # ------------------------------------------------------------------
    # Business totals
    # ------------------------------------------------------------------

    revenue_items = cleaned["revenue_items"]
    category_items = cleaned["category_items"]
    region_items = cleaned["region_items"]

    add_metric(
        "approved_revenue",
        round(
            float(
                revenue_items["item_revenue"].sum()
            ),
            2,
        ),
        stage="business_rules",
    )

    add_metric(
        "category_analysis_revenue",
        round(
            float(
                category_items["item_revenue"].sum()
            ),
            2,
        ),
        stage="business_rules",
    )

    add_metric(
        "region_analysis_revenue",
        round(
            float(
                region_items["item_revenue"].sum()
            ),
            2,
        ),
        stage="business_rules",
    )

    add_metric(
        "business_category_count",
        category_items[
            "business_category"
        ].nunique(),
        stage="business_rules",
    )

    add_metric(
        "region_count",
        region_items[
            "region"
        ].nunique(),
        stage="business_rules",
    )

    add_metric(
        "model_start",
        MODEL_START.date().isoformat(),
        stage="business_rules",
    )

    add_metric(
        "model_end",
        MODEL_END.date().isoformat(),
        stage="business_rules",
    )

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    for metric, value in feature_metrics.items():
        if isinstance(value, pd.Timestamp):
            value = value.isoformat()

        add_metric(
            metric,
            value,
            stage="feature_engineering",
        )

    for metric, value in reconciliation.items():
        status = (
            "PASS"
            if abs(value) <= 1e-6
            else "FAIL"
        )

        add_metric(
            metric,
            value,
            status=status,
            stage="reconciliation",
        )

    add_metric(
        "processed_dataset_exists",
        PROCESSED_DATASET_PATH.is_file(),
        stage="storage",
    )

    add_metric(
        "feature_dataset_exists",
        FEATURE_DATASET_PATH.is_file(),
        stage="storage",
    )

    return pd.DataFrame(rows)


def save_quality_report(
    report: pd.DataFrame,
) -> None:
    """Persist the automated data-quality report."""
    report.to_csv(
        DATA_QUALITY_REPORT_PATH,
        index=False,
    )


# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

def run_pipeline() -> dict[str, object]:
    """
    Execute the complete Data Layer pipeline.

    Returns
    -------
    dict[str, object]
        Paths and validation outputs from the completed run.
    """
    print("=== DATA LAYER PIPELINE ===")

    ensure_output_directories()

    # ------------------------------------------------------------------
    # 1. INGESTION
    # ------------------------------------------------------------------

    print("\n[1/6] Loading raw data...")

    raw_tables = load_raw_data()

    print(
        f"Loaded {len(raw_tables)} raw tables."
    )

    # ------------------------------------------------------------------
    # 2. CLEANING
    # ------------------------------------------------------------------

    print("\n[2/6] Cleaning and validating data...")

    cleaned = clean_data(
        raw_tables
    )

    cleaning_metrics = validate_cleaned_data(
        cleaned["revenue_items"],
        cleaned["category_items"],
        cleaned["region_items"],
    )

    print("Cleaning and validation passed.")

    # ------------------------------------------------------------------
    # 3. PROCESSED STORAGE
    # ------------------------------------------------------------------

    print("\n[3/6] Saving processed dataset...")

    processed = build_processed_dataset(
        cleaned
    )

    save_processed_dataset(
        processed
    )

    print(
        f"Saved: {PROCESSED_DATASET_PATH}"
    )

    # ------------------------------------------------------------------
    # 4. FEATURE ENGINEERING
    # ------------------------------------------------------------------

    print("\n[4/6] Building forecasting features...")

    features = build_feature_dataset(
        cleaned
    )

    feature_metrics = validate_feature_dataset(
        features
    )

    reconciliation = validate_target_reconciliation(
        features,
        cleaned,
    )

    print("Feature engineering and reconciliation passed.")

    # ------------------------------------------------------------------
    # 5. FEATURE STORAGE
    # ------------------------------------------------------------------

    print("\n[5/6] Saving feature dataset...")

    save_feature_dataset(
        features
    )

    print(
        f"Saved: {FEATURE_DATASET_PATH}"
    )

    # ------------------------------------------------------------------
    # 6. DATA QUALITY REPORT
    # ------------------------------------------------------------------

    print("\n[6/6] Generating data-quality report...")

    report = build_quality_report(
        raw_tables=raw_tables,
        cleaned=cleaned,
        features=features,
        cleaning_metrics=cleaning_metrics,
        feature_metrics=feature_metrics,
        reconciliation=reconciliation,
    )

    save_quality_report(
        report
    )

    print(
        f"Saved: {DATA_QUALITY_REPORT_PATH}"
    )

    print("\n=== PIPELINE COMPLETED ===")

    return {
        "processed_dataset": PROCESSED_DATASET_PATH,
        "feature_dataset": FEATURE_DATASET_PATH,
        "data_quality_report": DATA_QUALITY_REPORT_PATH,
        "cleaning_metrics": cleaning_metrics,
        "feature_metrics": feature_metrics,
        "reconciliation": reconciliation,
    }


# ============================================================================
# STANDALONE SMOKE TEST
# ============================================================================

def main() -> None:
    """Run the complete Data Layer pipeline."""
    result = run_pipeline()

    print("\n=== FINAL VALIDATION ===")

    print(
        "Processed dataset:",
        result["processed_dataset"],
    )

    print(
        "Feature dataset:",
        result["feature_dataset"],
    )

    print(
        "Data-quality report:",
        result["data_quality_report"],
    )

    print(
        "\nTarget reconciliation:"
    )

    for key, value in result[
        "reconciliation"
    ].items():
        print(
            f"  {key}: {value:.6f}"
        )


if __name__ == "__main__":
    main()