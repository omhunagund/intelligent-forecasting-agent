"""
Reproducibility Orchestrator
============================

Rebuild the generated artifacts required by the
Intelligent Forecasting Agent from the raw Olist dataset.

This is a full regeneration workflow. It executes the project's
existing Data, ML, production-forecast, explainability, monitoring,
knowledge-base, RAG, and reporting pipelines in dependency order.

Running this script may overwrite previously generated artifacts
and creates a new timestamped weekly business-intelligence report.

Prerequisites
-------------
1. Python dependencies installed.
2. A valid .env file with GROQ_API_KEY.
3. All nine required Olist CSV files present in data/raw/.

Execution
---------
Run from the project root:

    python -m scripts.reproduce_project
"""

from __future__ import annotations

from pathlib import Path

from src.data_layer.pipeline import run_pipeline
from src.ml_layer.training import run_stage1_training
from src.ml_layer.secondary_training import run_secondary_training
from src.ai_layer.production_forecast import run_production_forecast
from src.ml_layer.explainability import run_explainability
from src.ml_layer.monitoring import run_monitoring
from src.ai_layer.knowledge_base import run_knowledge_base_generation
from src.ai_layer.rag_index import run_rag_indexing
from src.ai_layer.weekly_report import run_weekly_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_RAW_FILES = [
    PROJECT_ROOT
    / "data"
    / "raw"
    / "olist_customers_dataset.csv",

    PROJECT_ROOT
    / "data"
    / "raw"
    / "olist_geolocation_dataset.csv",

    PROJECT_ROOT
    / "data"
    / "raw"
    / "olist_order_items_dataset.csv",

    PROJECT_ROOT
    / "data"
    / "raw"
    / "olist_order_payments_dataset.csv",

    PROJECT_ROOT
    / "data"
    / "raw"
    / "olist_order_reviews_dataset.csv",

    PROJECT_ROOT
    / "data"
    / "raw"
    / "olist_orders_dataset.csv",

    PROJECT_ROOT
    / "data"
    / "raw"
    / "olist_products_dataset.csv",

    PROJECT_ROOT
    / "data"
    / "raw"
    / "olist_sellers_dataset.csv",

    PROJECT_ROOT
    / "data"
    / "raw"
    / "product_category_name_translation.csv",
]


REQUIRED_ARTIFACTS = [
    # Data layer
    PROJECT_ROOT
    / "data"
    / "processed"
    / "processed_dataset.parquet",

    PROJECT_ROOT
    / "data"
    / "features"
    / "forecasting_features.parquet",

    PROJECT_ROOT
    / "reports"
    / "data_quality"
    / "data_quality_report.csv",

    # Overall ML
    PROJECT_ROOT
    / "reports"
    / "stage1_overall_model_comparison.csv",

    PROJECT_ROOT
    / "reports"
    / "stage1_weighted_ensemble_predictions.csv",

    # Secondary ML
    PROJECT_ROOT
    / "reports"
    / "secondary"
    / "secondary_latest_forecasts.csv",

    PROJECT_ROOT
    / "reports"
    / "secondary"
    / "secondary_predictions.parquet",

    # Production model + forecast
    PROJECT_ROOT
    / "src"
    / "ml_layer"
    / "models"
    / "overall_production_xgboost.json",

    PROJECT_ROOT
    / "reports"
    / "stage1_overall_production_xgboost_forecast.csv",

    # SHAP
    PROJECT_ROOT
    / "reports"
    / "shap"
    / "agent_shap_explanations.json",

    PROJECT_ROOT
    / "reports"
    / "shap"
    / "global_feature_importance.csv",

    PROJECT_ROOT
    / "reports"
    / "shap"
    / "overall_local_explanations.csv",

    PROJECT_ROOT
    / "reports"
    / "shap"
    / "secondary_local_explanations.csv",

    PROJECT_ROOT
    / "reports"
    / "shap"
    / "shap_summary.csv",

    PROJECT_ROOT
    / "reports"
    / "shap"
    / "top_forecast_drivers.csv",

    # Monitoring
    PROJECT_ROOT
    / "reports"
    / "monitoring"
    / "data_drift_report.csv",

    PROJECT_ROOT
    / "reports"
    / "monitoring"
    / "model_performance_report.csv",

    PROJECT_ROOT
    / "reports"
    / "monitoring"
    / "monitoring_summary.json",

    # Knowledge base
    PROJECT_ROOT
    / "data"
    / "knowledge_base"
    / "business_context"
    / "overall_forecast_analysis.md",

    PROJECT_ROOT
    / "data"
    / "knowledge_base"
    / "business_context"
    / "category_forecast_analysis.md",

    PROJECT_ROOT
    / "data"
    / "knowledge_base"
    / "business_context"
    / "regional_forecast_analysis.md",

    PROJECT_ROOT
    / "data"
    / "knowledge_base"
    / "business_context"
    / "explainability_context.md",

    PROJECT_ROOT
    / "data"
    / "knowledge_base"
    / "business_context"
    / "monitoring_context.md",

    PROJECT_ROOT
    / "data"
    / "knowledge_base"
    / "business_context"
    / "data_quality_context.md",

    PROJECT_ROOT
    / "data"
    / "knowledge_base"
    / "forecast_history"
    / "historical_forecast_outcomes.md",
]


def validate_raw_dataset() -> None:
    """Verify that all required Olist CSV files are present."""

    print()
    print("=" * 72)
    print("RAW DATASET VALIDATION")
    print("=" * 72)

    missing: list[Path] = []

    for path in REQUIRED_RAW_FILES:
        if not path.is_file():
            missing.append(path)

    if missing:
        print()
        print("[FAIL] Required Olist dataset files are missing:")
        print()

        for path in missing:
            print(
                f"- {path.relative_to(PROJECT_ROOT)}"
            )

        print()
        print(
            "Download the Olist dataset and place all required CSV "
            "files in data/raw/ before running the reproducibility "
            "workflow."
        )

        raise RuntimeError(
            "Raw Olist dataset validation failed."
        )

    print(
        f"[PASS] Verified all "
        f"{len(REQUIRED_RAW_FILES)} required Olist CSV files."
    )


def run_stage(
    stage_name: str,
    stage_function,
) -> None:
    """Run one project stage with a clear progress message."""

    print()
    print("=" * 72)
    print(stage_name)
    print("=" * 72)

    try:
        stage_function()
    except Exception as exc:
        raise RuntimeError(
            f"Reproducibility stage failed: {stage_name}"
        ) from exc

    print(f"[PASS] {stage_name}")


def validate_required_artifacts() -> None:
    """Verify that all required generated artifacts exist and are non-empty."""

    print()
    print("=" * 72)
    print("FINAL ARTIFACT VALIDATION")
    print("=" * 72)

    missing: list[Path] = []
    empty: list[Path] = []

    for path in REQUIRED_ARTIFACTS:

        if not path.is_file():
            missing.append(path)
            continue

        if path.stat().st_size == 0:
            empty.append(path)

    # Weekly reports are timestamped, so validate them separately.
    weekly_report_dir = (
        PROJECT_ROOT
        / "reports"
        / "weekly_reports"
    )

    weekly_reports = sorted(
        weekly_report_dir.glob(
            "weekly_business_intelligence_*.md"
        )
    )

    if not weekly_reports:
        missing.append(
            weekly_report_dir
            / "weekly_business_intelligence_*.md"
        )
    else:
        latest_report = weekly_reports[-1]

        if latest_report.stat().st_size == 0:
            empty.append(latest_report)

    # ChromaDB uses a directory rather than one fixed file.
    vector_store_dir = (
        PROJECT_ROOT
        / "data"
        / "vector_store"
    )

    if not vector_store_dir.is_dir():
        missing.append(vector_store_dir)

    if missing:
        print()
        print("[FAIL] Missing required artifacts:")

        for path in missing:
            try:
                display_path = path.relative_to(
                    PROJECT_ROOT
                )
            except ValueError:
                display_path = path

            print(f"- {display_path}")

    if empty:
        print()
        print("[FAIL] Empty required artifacts:")

        for path in empty:
            print(
                f"- {path.relative_to(PROJECT_ROOT)}"
            )

    if missing or empty:
        raise RuntimeError(
            "Reproduction completed with invalid or missing artifacts."
        )

    print(
        f"[PASS] Verified "
        f"{len(REQUIRED_ARTIFACTS)} fixed artifacts."
    )

    print(
        f"[PASS] Verified "
        f"{len(weekly_reports)} weekly report(s)."
    )

    print(
        "[PASS] Verified ChromaDB vector-store directory."
    )


def main() -> None:
    """Run the complete reproducibility workflow."""

    print("=" * 72)
    print("INTELLIGENT FORECASTING AGENT")
    print("REPRODUCIBILITY WORKFLOW")
    print("=" * 72)

    print()
    print(
        "Project root:"
        f"\n{PROJECT_ROOT}"
    )

    validate_raw_dataset()

    run_stage(
        "1/9 Data Layer",
        run_pipeline,
    )

    run_stage(
        "2/9 Overall ML Training",
        run_stage1_training,
    )

    run_stage(
        "3/9 Secondary ML Training",
        run_secondary_training,
    )

    run_stage(
        "4/9 Production Overall Forecast",
        run_production_forecast,
    )

    run_stage(
        "5/9 TreeSHAP Explainability",
        run_explainability,
    )

    run_stage(
        "6/9 Monitoring",
        run_monitoring,
    )

    run_stage(
        "7/9 Knowledge Base Generation",
        run_knowledge_base_generation,
    )

    run_stage(
        "8/9 RAG Index Generation",
        run_rag_indexing,
    )

    run_stage(
        "9/9 Weekly Business Report",
        run_weekly_report,
    )

    validate_required_artifacts()

    print()
    print("=" * 72)
    print("REPRODUCIBILITY WORKFLOW COMPLETED SUCCESSFULLY")
    print("=" * 72)


if __name__ == "__main__":
    main()