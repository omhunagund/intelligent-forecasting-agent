"""
Forecast Risk Assessment
========================

Deterministic risk assessment for the AI Decision Layer.

Inputs
------
1. Latest production forecast
2. Recent model-performance monitoring
3. Data-drift monitoring
4. Forecast interval width

Scoring
-------
Project-defined scoring only; these are NOT industry-standard
thresholds.

Performance component:
    stable  ->  0 points
    warning -> 25 points
    alert   -> 50 points

Drift component:
    each alerting drift feature contributes proportionally
    up to 30 points.

Forecast uncertainty component:
    interval width relative to forecast contributes up to
    20 points.

Final risk status:
    0–24   -> stable
    25–49  -> warning
    50–100 -> alert

No LLM logic is used here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

PERFORMANCE_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "monitoring"
    / "model_performance_report.csv"
)

DRIFT_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "monitoring"
    / "data_drift_report.csv"
)


# ============================================================================
# PROJECT-DEFINED RISK CONFIGURATION
# ============================================================================

PERFORMANCE_MAX_POINTS = 50
DRIFT_MAX_POINTS = 30
UNCERTAINTY_MAX_POINTS = 20

STABLE_MAX_SCORE = 24
WARNING_MAX_SCORE = 49

# Project-defined interval-width thresholds.
UNCERTAINTY_WARNING_RATIO = 0.50
UNCERTAINTY_ALERT_RATIO = 0.75

# Performance status mapping.
PERFORMANCE_STATUS_POINTS = {
    "stable": 0.0,
    "warning": 25.0,
    "alert": 50.0,
}


# ============================================================================
# LOAD MONITORING REPORTS
# ============================================================================

def load_performance_report() -> pd.DataFrame:
    """Load model-performance monitoring results."""

    if not PERFORMANCE_REPORT_PATH.is_file():
        raise FileNotFoundError(
            "Model performance monitoring report not found:\n"
            f"{PERFORMANCE_REPORT_PATH}"
        )

    df = pd.read_csv(
        PERFORMANCE_REPORT_PATH
    )

    required_columns = {
        "series_type",
        "series_id",
        "baseline_mae",
        "baseline_rmse",
        "baseline_mape",
        "recent_mae",
        "recent_rmse",
        "recent_mape",
        "status",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Model performance report is missing "
            f"columns: {sorted(missing)}"
        )

    return df


def load_drift_report() -> pd.DataFrame:
    """Load data-drift monitoring results."""

    if not DRIFT_REPORT_PATH.is_file():
        raise FileNotFoundError(
            "Data-drift monitoring report not found:\n"
            f"{DRIFT_REPORT_PATH}"
        )

    df = pd.read_csv(
        DRIFT_REPORT_PATH
    )

    required_columns = {
        "series_type",
        "series_id",
        "feature",
        "ks_statistic",
        "ks_p_value",
        "psi",
        "status",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Data-drift report is missing "
            f"columns: {sorted(missing)}"
        )

    return df


# ============================================================================
# SERIES MONITORING LOOKUPS
# ============================================================================

def get_performance_status(
    series_type: str,
    series_id: str,
    performance_df: pd.DataFrame,
) -> dict:
    """Return monitoring performance for one series."""

    matching = performance_df[
        (
            performance_df["series_type"]
            == series_type
        )
        & (
            performance_df["series_id"]
            == series_id
        )
    ]

    if matching.empty:
        raise ValueError(
            f"No performance monitoring record found for "
            f"{series_type}/{series_id}."
        )

    row = matching.iloc[0]

    return {
        "status":
            str(
                row["status"]
            ).lower(),
        "baseline_mae":
            float(
                row["baseline_mae"]
            ),
        "baseline_rmse":
            float(
                row["baseline_rmse"]
            ),
        "baseline_mape":
            float(
                row["baseline_mape"]
            ),
        "recent_mae":
            float(
                row["recent_mae"]
            ),
        "recent_rmse":
            float(
                row["recent_rmse"]
            ),
        "recent_mape":
            float(
                row["recent_mape"]
            ),
        "baseline_start":
            str(
                row.get(
                    "baseline_start",
                    "",
                )
            ),
        "baseline_end":
            str(
                row.get(
                    "baseline_end",
                    "",
                )
            ),
        "recent_start":
            str(
                row.get(
                    "recent_start",
                    "",
                )
            ),
        "recent_end":
            str(
                row.get(
                    "recent_end",
                    "",
                )
            ),
    }


def get_drift_status(
    series_type: str,
    series_id: str,
    drift_df: pd.DataFrame,
) -> dict:
    """Return aggregated drift monitoring results for one series."""

    matching = drift_df[
        (
            drift_df["series_type"]
            == series_type
        )
        & (
            drift_df["series_id"]
            == series_id
        )
    ].copy()

    if matching.empty:
        raise ValueError(
            f"No drift monitoring record found for "
            f"{series_type}/{series_id}."
        )

    alert_features = matching[
        matching["status"]
        .astype(str)
        .str.lower()
        == "alert"
    ]

    warning_features = matching[
        matching["status"]
        .astype(str)
        .str.lower()
        == "warning"
    ]

    stable_features = matching[
        matching["status"]
        .astype(str)
        .str.lower()
        == "stable"
    ]

    total_features = len(
        matching
    )

    return {
        "status":
            (
                "alert"
                if len(alert_features) > 0
                else (
                    "warning"
                    if len(warning_features) > 0
                    else "stable"
                )
            ),
        "total_features":
            total_features,
        "alert_features":
            int(
                len(
                    alert_features
                )
            ),
        "warning_features":
            int(
                len(
                    warning_features
                )
            ),
        "stable_features":
            int(
                len(
                    stable_features
                )
            ),
        "features":
            matching[
                [
                    "feature",
                    "ks_statistic",
                    "ks_p_value",
                    "psi",
                    "status",
                ]
            ].to_dict(
                orient="records"
            ),
    }


# ============================================================================
# RISK COMPONENTS
# ============================================================================

def calculate_performance_risk(
    performance_status: str,
) -> float:
    """Convert monitoring performance status to risk points."""

    status = performance_status.lower()

    if status not in PERFORMANCE_STATUS_POINTS:
        raise ValueError(
            f"Unknown performance status: {performance_status}"
        )

    return PERFORMANCE_STATUS_POINTS[
        status
    ]


def calculate_drift_risk(
    drift_info: dict,
) -> float:
    """
    Convert feature-level drift results into 0–30 project-defined points.

    Alerting drift features contribute 100% of their proportional
    share of the 30-point drift budget.

    Warning features contribute 50% of their proportional share.
    """

    total_features = int(
        drift_info["total_features"]
    )

    if total_features <= 0:
        return 0.0

    alert_features = float(
        drift_info["alert_features"]
    )

    warning_features = float(
        drift_info["warning_features"]
    )

    weighted_drift = (
        alert_features
        + 0.5 * warning_features
    )

    score = (
        weighted_drift
        / total_features
        * DRIFT_MAX_POINTS
    )

    return float(
        min(
            score,
            DRIFT_MAX_POINTS,
        )
    )


def calculate_uncertainty_risk(
    forecast_revenue: float,
    lower_80: float,
    upper_80: float,
) -> tuple[float, float]:
    """
    Calculate interval uncertainty risk.

    Returns
    -------
    score:
        0–20 project-defined risk points.

    interval_ratio:
        interval width / forecast value.
    """

    if forecast_revenue <= 0:
        raise ValueError(
            "forecast_revenue must be positive "
            "for uncertainty assessment."
        )

    if lower_80 > upper_80:
        raise ValueError(
            "lower_80 cannot exceed upper_80."
        )

    interval_width = (
        upper_80
        - lower_80
    )

    interval_ratio = (
        interval_width
        / forecast_revenue
    )

    if (
        interval_ratio
        >= UNCERTAINTY_ALERT_RATIO
    ):
        score = UNCERTAINTY_MAX_POINTS

    elif (
        interval_ratio
        >= UNCERTAINTY_WARNING_RATIO
    ):
        # Linear interpolation between 10 and 20 points.
        score = (
            UNCERTAINTY_MAX_POINTS
            * (
                interval_ratio
                / UNCERTAINTY_ALERT_RATIO
            )
        )

        score = max(
            10.0,
            min(
                score,
                UNCERTAINTY_MAX_POINTS,
            ),
        )

    else:
        # Below warning threshold.
        score = (
            UNCERTAINTY_MAX_POINTS
            * (
                interval_ratio
                / UNCERTAINTY_WARNING_RATIO
            )
        )

        score = max(
            0.0,
            min(
                score,
                10.0,
            ),
        )

    return (
        float(score),
        float(interval_ratio),
    )


def classify_risk(
    score: float,
) -> str:
    """Convert total project-defined score to a status."""

    if score <= STABLE_MAX_SCORE:
        return "stable"

    if score <= WARNING_MAX_SCORE:
        return "warning"

    return "alert"


# ============================================================================
# REASON GENERATION
# ============================================================================

def build_risk_reasons(
    performance: dict,
    drift: dict,
    uncertainty_ratio: float,
) -> list[str]:
    """Generate factual, project-derived reasons for the risk status."""

    reasons: list[str] = []

    performance_status = (
        performance["status"]
    )

    if performance_status == "alert":

        reasons.append(
            "Recent model performance is in alert status."
        )

        reasons.append(
            "Recent MAPE is "
            f"{performance['recent_mape']:.2f}% "
            f"versus a baseline MAPE of "
            f"{performance['baseline_mape']:.2f}%."
        )

    elif performance_status == "warning":

        reasons.append(
            "Recent model performance is in warning status."
        )

        reasons.append(
            "Recent MAPE is "
            f"{performance['recent_mape']:.2f}% "
            f"versus a baseline MAPE of "
            f"{performance['baseline_mape']:.2f}%."
        )

    else:

        reasons.append(
            "Recent model performance is currently stable "
            "under the project's monitoring rules."
        )

    if drift["alert_features"] > 0:

        reasons.append(
            f"{drift['alert_features']} of "
            f"{drift['total_features']} monitored "
            "features have drift status 'alert'."
        )

    elif drift["warning_features"] > 0:

        reasons.append(
            f"{drift['warning_features']} of "
            f"{drift['total_features']} monitored "
            "features have drift status 'warning'."
        )

    else:

        reasons.append(
            "No monitored features currently have "
            "warning or alert drift status."
        )

    if (
        uncertainty_ratio
        >= UNCERTAINTY_ALERT_RATIO
    ):

        reasons.append(
            "The 80% forecast interval is wide relative "
            "to the forecast value under the project's "
            "uncertainty rule."
        )

    elif (
        uncertainty_ratio
        >= UNCERTAINTY_WARNING_RATIO
    ):

        reasons.append(
            "The 80% forecast interval shows elevated "
            "uncertainty under the project's uncertainty rule."
        )

    else:

        reasons.append(
            "The forecast interval is below the project's "
            "uncertainty-warning threshold."
        )

    return reasons


def build_confidence_note(
    risk_status: str,
    performance: dict,
    drift: dict,
    uncertainty_ratio: float,
) -> str:
    """Create a transparent confidence statement."""

    if risk_status == "alert":

        return (
            "Forecast reliability should be treated with caution. "
            "The project monitoring outputs contain one or more "
            "strong risk signals."
        )

    if risk_status == "warning":

        return (
            "Forecast reliability has some concerns under the "
            "project-defined monitoring rules. Results should be "
            "reviewed alongside the supporting evidence."
        )

    return (
        "No major reliability concern was detected under the "
        "project-defined monitoring rules."
    )


# ============================================================================
# PUBLIC FUNCTION
# ============================================================================

def assess_forecast_risk_data(
    series_type: str,
    series_id: str,
    forecast_revenue: float,
    lower_80: float,
    upper_80: float,
) -> dict:
    """
    Produce a deterministic risk assessment for one forecast.

    All evidence is derived from saved project outputs.
    """

    if series_type not in {
        "overall",
        "category",
        "region",
    }:
        raise ValueError(
            "Invalid series_type."
        )

    if not series_id.strip():
        raise ValueError(
            "series_id must not be empty."
        )

    if (
        series_type == "overall"
        and series_id != "overall"
    ):
        raise ValueError(
            "For series_type='overall', "
            "series_id must be 'overall'."
        )

    if forecast_revenue <= 0:
        raise ValueError(
            "forecast_revenue must be positive."
        )

    if lower_80 > upper_80:
        raise ValueError(
            "lower_80 cannot exceed upper_80."
        )

    performance_df = (
        load_performance_report()
    )

    drift_df = (
        load_drift_report()
    )

    performance = get_performance_status(
        series_type=series_type,
        series_id=series_id,
        performance_df=performance_df,
    )

    drift = get_drift_status(
        series_type=series_type,
        series_id=series_id,
        drift_df=drift_df,
    )

    performance_score = (
        calculate_performance_risk(
            performance[
                "status"
            ]
        )
    )

    drift_score = (
        calculate_drift_risk(
            drift
        )
    )

    uncertainty_score, interval_ratio = (
        calculate_uncertainty_risk(
            forecast_revenue=forecast_revenue,
            lower_80=lower_80,
            upper_80=upper_80,
        )
    )

    total_score = (
        performance_score
        + drift_score
        + uncertainty_score
    )

    total_score = float(
        min(
            total_score,
            100.0,
        )
    )

    status = classify_risk(
        total_score
    )

    reasons = build_risk_reasons(
        performance=performance,
        drift=drift,
        uncertainty_ratio=interval_ratio,
    )

    confidence_note = (
        build_confidence_note(
            risk_status=status,
            performance=performance,
            drift=drift,
            uncertainty_ratio=interval_ratio,
        )
    )

    return {
        "series_type":
            series_type,
        "series_id":
            series_id,
        "status":
            status,
        "score":
            round(
                total_score,
                2,
            ),
        "risk_components":
            {
                "performance":
                    round(
                        performance_score,
                        2,
                    ),
                "drift":
                    round(
                        drift_score,
                        2,
                    ),
                "uncertainty":
                    round(
                        uncertainty_score,
                        2,
                    ),
            },
        "drift_status":
            drift[
                "status"
            ],
        "performance_status":
            performance[
                "status"
            ],
        "interval_width":
            float(
                upper_80
                - lower_80
            ),
        "interval_ratio":
            round(
                interval_ratio,
                4,
            ),
        "recent_mape":
            performance[
                "recent_mape"
            ],
        "baseline_mape":
            performance[
                "baseline_mape"
            ],
        "drift_alert_features":
            drift[
                "alert_features"
            ],
        "drift_warning_features":
            drift[
                "warning_features"
            ],
        "total_monitored_features":
            drift[
                "total_features"
            ],
        "reasons":
            reasons,
        "confidence_note":
            confidence_note,
        "method":
            "project_defined_risk_scoring_v1",
    }