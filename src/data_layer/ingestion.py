"""
Raw data ingestion for the Intelligent Business Forecasting Agent.

Responsibilities
----------------
1. Locate the project root reliably.
2. Locate the immutable raw Olist dataset directory.
3. Verify that all required source files exist.
4. Load the raw CSV files into pandas DataFrames.
5. Return the loaded tables in a consistent dictionary.

Important
---------
This module does NOT:
- modify raw files,
- clean data,
- apply business rules,
- remove records,
- create forecasting features,
- write processed datasets.

Those responsibilities belong to later Data Layer modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
RAW_DATA_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------------------------
# Required raw Olist files
# ---------------------------------------------------------------------------

RAW_FILES: Final[dict[str, str]] = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """Return the absolute project root directory."""
    return PROJECT_ROOT


def get_raw_data_dir() -> Path:
    """Return the directory containing the immutable raw datasets."""
    return RAW_DATA_DIR


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_raw_files(
    raw_data_dir: Path | None = None,
) -> None:
    """
    Verify that every required raw dataset file exists.

    Raises
    ------
    FileNotFoundError
        If one or more required files are missing.
    """
    data_dir = (
        Path(raw_data_dir)
        if raw_data_dir is not None
        else RAW_DATA_DIR
    )

    missing_files = [
        filename
        for filename in RAW_FILES.values()
        if not (data_dir / filename).is_file()
    ]

    if missing_files:
        missing_text = "\n".join(
            f"  - {filename}"
            for filename in missing_files
        )

        raise FileNotFoundError(
            "Required raw dataset files are missing:\n"
            f"{missing_text}\n\n"
            f"Expected directory: {data_dir}"
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw_data(
    raw_data_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load all required Olist raw CSV files.

    Parameters
    ----------
    raw_data_dir:
        Optional custom raw-data directory.
        Defaults to ``data/raw`` in the project root.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary containing all nine Olist tables.

    Notes
    -----
    The source CSV files are read only.
    No modifications are made to the raw files.
    """
    data_dir = (
        Path(raw_data_dir)
        if raw_data_dir is not None
        else RAW_DATA_DIR
    )

    validate_raw_files(data_dir)

    tables: dict[str, pd.DataFrame] = {}

    for table_name, filename in RAW_FILES.items():
        file_path = data_dir / filename

        tables[table_name] = pd.read_csv(
            file_path,
            low_memory=False,
        )

    return tables


# ---------------------------------------------------------------------------
# Basic ingestion summary
# ---------------------------------------------------------------------------

def summarize_loaded_data(
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Return a compact summary of loaded raw tables.

    This is a read-only diagnostic helper and does not modify
    any DataFrame or raw source file.
    """
    summary = []

    for table_name, dataframe in tables.items():
        summary.append(
            {
                "table": table_name,
                "rows": len(dataframe),
                "columns": len(dataframe.columns),
                "missing_cells": int(
                    dataframe.isna().sum().sum()
                ),
            }
        )

    return pd.DataFrame(summary)


# ---------------------------------------------------------------------------
# Module smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== RAW DATA INGESTION TEST ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw data directory: {RAW_DATA_DIR}")

    loaded_tables = load_raw_data()

    print(
        f"\nSuccessfully loaded "
        f"{len(loaded_tables)} raw tables."
    )

    print("\n=== TABLE SUMMARY ===")
    print(
        summarize_loaded_data(loaded_tables)
        .to_string(index=False)
    )