"""
Streamlit Serving & Interface Layer
====================================

Two-panel dashboard required by the project blueprint:

Panel 1
-------
Forecast Dashboard
- Historical actuals
- Forecasts
- 80% prediction intervals
- TreeSHAP drivers
- Monitoring / drift status
- Same-period-last-year comparison

Panel 2
-------
AI Business Intelligence
- Natural-language query box
- Agent response
- Sources
- Latest weekly report
- Downloadable PDF report
"""

from __future__ import annotations

import io
import re
import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "forecasting_features.parquet"
)

MONITORING_DRIFT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "monitoring"
    / "data_drift_report.csv"
)

MONITORING_PERFORMANCE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "monitoring"
    / "model_performance_report.csv"
)


# ============================================================================
# STREAMLIT CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Intelligent Forecasting Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CONSTANTS
# ============================================================================

CATEGORY_IDS = [
    "Automotive",
    "Beauty & Health",
    "Books & Media",
    "Electronics & Computing",
    "Fashion & Accessories",
    "Food & Beverage",
    "Gifts, Arts & Seasonal",
    "Home & Furniture",
    "Home Improvement & Garden",
    "Kids & Baby",
    "Kitchen & Appliances",
    "Office, Business & Services",
    "Pet Supplies",
    "Phones & Telecom",
    "Sports & Leisure",
]

REGION_IDS = [
    "Central-West",
    "North",
    "Northeast",
    "South",
    "Southeast",
]


# ============================================================================
# API HELPERS
# ============================================================================

def api_get(
    base_url: str,
    endpoint: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:

    url = (
        base_url.rstrip("/")
        + endpoint
    )

    response = requests.get(
        url,
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    return response.json()


def api_post(
    base_url: str,
    endpoint: str,
    payload: dict[str, Any],
) -> dict[str, Any]:

    url = (
        base_url.rstrip("/")
        + endpoint
    )

    response = requests.post(
        url,
        json=payload,
        timeout=180,
    )

    response.raise_for_status()

    return response.json()


# ============================================================================
# DATA HELPERS
# ============================================================================

@st.cache_data
def load_feature_data() -> pd.DataFrame:
    """Load the canonical project feature dataset."""

    if not FEATURE_DATA_PATH.is_file():

        raise FileNotFoundError(
            "Canonical feature dataset not found:\n"
            f"{FEATURE_DATA_PATH}"
        )

    data = pd.read_parquet(
        FEATURE_DATA_PATH
    )

    required_columns = {
        "timestamp",
        "series_type",
        "series_id",
        "target_revenue",
    }

    missing = (
        required_columns
        - set(data.columns)
    )

    if missing:

        raise ValueError(
            "Feature dataset is missing required "
            f"columns: {sorted(missing)}"
        )

    data = data.copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )

    data["target_revenue"] = pd.to_numeric(
        data["target_revenue"],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            "timestamp",
            "target_revenue",
        ]
    )

    return data.sort_values(
        "timestamp"
    )


@st.cache_data
def load_drift_data() -> pd.DataFrame | None:
    """Load detailed drift monitoring data."""

    if not MONITORING_DRIFT_PATH.is_file():
        return None

    try:

        data = pd.read_csv(
            MONITORING_DRIFT_PATH
        )

        return data

    except Exception:
        return None


@st.cache_data
def load_performance_data() -> pd.DataFrame | None:
    """Load detailed model-performance monitoring data."""

    if not MONITORING_PERFORMANCE_PATH.is_file():
        return None

    try:

        data = pd.read_csv(
            MONITORING_PERFORMANCE_PATH
        )

        return data

    except Exception:
        return None


def filter_series(
    data: pd.DataFrame,
    series_type: str,
    series_id: str,
) -> pd.DataFrame:

    return (
        data.loc[
            (data["series_type"] == series_type)
            & (data["series_id"] == series_id)
        ]
        .sort_values("timestamp")
        .copy()
    )


# ============================================================================
# FORMATTING HELPERS
# ============================================================================

def money(
    value: float | int | None,
) -> str:

    if value is None:
        return "N/A"

    return f"${value:,.2f}"


def risk_status_label(
    status: str | None,
) -> str:

    if not status:
        return "UNKNOWN"

    return str(status).upper()


def status_emoji(
    status: str | None,
) -> str:

    normalized = (
        str(status).lower()
        if status
        else ""
    )

    if normalized == "stable":
        return "🟢"

    if normalized == "warning":
        return "🟡"

    if normalized == "alert":
        return "🔴"

    return "⚪"


def make_pdf(
    markdown_text: str,
) -> bytes:
    """
    Convert the report's Markdown into a simple PDF.

    The report content is intentionally kept text-centric so the
    downloaded artifact remains lightweight and portable.
    """

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    body_style = styles["BodyText"]

    story = []

    for raw_line in markdown_text.splitlines():

        line = raw_line.strip()

        if not line:
            story.append(
                Spacer(
                    1,
                    8,
                )
            )
            continue

        if line.startswith("# "):

            text = re.sub(
                r"^#\s+",
                "",
                line,
            )

            story.append(
                Paragraph(
                    text,
                    title_style,
                )
            )

        elif line.startswith("## "):

            text = re.sub(
                r"^##\s+",
                "",
                line,
            )

            story.append(
                Paragraph(
                    text,
                    heading_style,
                )
            )

        elif line.startswith("### "):

            text = re.sub(
                r"^###\s+",
                "",
                line,
            )

            story.append(
                Paragraph(
                    text,
                    heading_style,
                )
            )

        else:

            text = line

            text = re.sub(
                r"\*\*(.*?)\*\*",
                r"<b>\1</b>",
                text,
            )

            text = re.sub(
                r"`(.*?)`",
                r"<font name='Courier'>\1</font>",
                text,
            )

            story.append(
                Paragraph(
                    text,
                    body_style,
                )
            )

    document.build(
        story
    )

    buffer.seek(0)

    return buffer.read()


# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title(
    "⚙️ Dashboard Settings"
)

api_url = st.sidebar.text_input(
    "FastAPI URL",
    value=os.getenv(
        "FASTAPI_URL",
        "http://127.0.0.1:8000",
    ),
)

st.sidebar.markdown(
    "---"
)

st.sidebar.caption(
    "Intelligent Forecasting Agent"
)

st.sidebar.caption(
    "ML Forecasting + SHAP + RAG + LangGraph"
)


# ============================================================================
# HEADER
# ============================================================================

st.title(
    "📈 Intelligent Forecasting Agent"
)

st.caption(
    "Forecasting, explainability, risk monitoring, and AI-powered "
    "business intelligence."
)


# ============================================================================
# API HEALTH
# ============================================================================

try:

    health = api_get(
        api_url,
        "/health",
    )

    monitoring_status = health.get(
        "monitoring_status"
    )

    st.sidebar.success(
        "FastAPI connected"
    )

    st.sidebar.metric(
        "Monitoring Status",
        risk_status_label(
            monitoring_status
        ),
    )

except Exception as exc:

    st.sidebar.error(
        "FastAPI unavailable"
    )

    st.sidebar.caption(
        str(exc)
    )

    st.error(
        "The dashboard cannot connect to the FastAPI backend. "
        "Start it with:\n\n"
        "`python -m uvicorn src.serving.api:app --reload`"
    )

    st.stop()


# ============================================================================
# LOAD LOCAL DATA
# ============================================================================

try:

    feature_data = load_feature_data()

except Exception as exc:

    st.error(
        f"Unable to load project data: {exc}"
    )

    st.stop()


# ============================================================================
# PANEL 1 — FORECAST DASHBOARD
# ============================================================================

st.header(
    "Panel 1 — Forecast Dashboard"
)

selection_col1, selection_col2, selection_col3 = (
    st.columns(3)
)


with selection_col1:

    selected_type = st.selectbox(
        "Series Type",
        options=[
            "overall",
            "category",
            "region",
        ],
        format_func=lambda value:
            value.title(),
    )


with selection_col2:

    if selected_type == "overall":

        selected_id = "overall"

        st.text_input(
            "Series",
            value="Overall",
            disabled=True,
        )

    elif selected_type == "category":

        selected_id = st.selectbox(
            "Category",
            CATEGORY_IDS,
        )

    else:

        selected_id = st.selectbox(
            "Region",
            REGION_IDS,
        )


with selection_col3:

    horizon = st.selectbox(
        "Forecast Horizon",
        options=[4, 8, 12],
        index=0,
        format_func=lambda value:
            f"{value} weeks",
    )


series_data = filter_series(
    feature_data,
    selected_type,
    selected_id,
)

if series_data.empty:

    st.warning(
        "No historical data is available for this series."
    )

else:

    # ------------------------------------------------------------------------
    # Forecast API
    # ------------------------------------------------------------------------

    try:

        forecast_result = api_post(
            api_url,
            "/forecast",
            {
                "series_type":
                    selected_type,
                "series_id":
                    selected_id,
                "horizon":
                    horizon,
            },
        )

    except Exception as exc:

        st.error(
            f"Forecast API error: {exc}"
        )

        forecast_result = None


    if forecast_result:

        forecasts = forecast_result.get(
            "forecasts",
            [],
        )

        forecast_df = pd.DataFrame(
            forecasts
        )

        if not forecast_df.empty:

            forecast_df["timestamp"] = (
                pd.to_datetime(
                    forecast_df["timestamp"]
                )
            )

        # --------------------------------------------------------------------
        # KPI ROW
        # --------------------------------------------------------------------

        kpi1, kpi2, kpi3, kpi4 = (
            st.columns(4)
        )

        latest_forecast_value = None
        latest_lower = None
        latest_upper = None
        latest_timestamp = None

        if not forecast_df.empty:

            latest = (
                forecast_df
                .sort_values("timestamp")
                .iloc[-1]
            )

            latest_forecast_value = float(
                latest["forecast_revenue"]
            )

            latest_lower = float(
                latest["lower_80"]
            )

            latest_upper = float(
                latest["upper_80"]
            )

            latest_timestamp = (
                latest["timestamp"]
            )

        with kpi1:

            st.metric(
                "Latest Forecast",
                money(
                    latest_forecast_value
                ),
            )

        with kpi2:

            st.metric(
                "80% Lower",
                money(
                    latest_lower
                ),
            )

        with kpi3:

            st.metric(
                "80% Upper",
                money(
                    latest_upper
                ),
            )

        with kpi4:

            st.metric(
                "Production Model",
                forecast_result.get(
                    "model",
                    "N/A",
                ),
            )

        # --------------------------------------------------------------------
        # FORECAST CHART
        # --------------------------------------------------------------------

        st.subheader(
            "Historical Revenue + Forecast"
        )

        historical_plot = (
            series_data[
                [
                    "timestamp",
                    "target_revenue",
                ]
            ]
            .tail(52)
            .copy()
        )

        figure = go.Figure()

        figure.add_trace(
            go.Scatter(
                x=historical_plot[
                    "timestamp"
                ],
                y=historical_plot[
                    "target_revenue"
                ],
                mode="lines+markers",
                name="Actual Revenue",
            )
        )

        if not forecast_df.empty:

            figure.add_trace(
                go.Scatter(
                    x=forecast_df[
                        "timestamp"
                    ],
                    y=forecast_df[
                        "forecast_revenue"
                    ],
                    mode="lines+markers",
                    name="Forecast",
                )
            )

            figure.add_trace(
                go.Scatter(
                    x=list(
                        forecast_df[
                            "timestamp"
                        ]
                    )
                    + list(
                        forecast_df[
                            "timestamp"
                        ][::-1]
                    ),
                    y=list(
                        forecast_df[
                            "upper_80"
                        ]
                    )
                    + list(
                        forecast_df[
                            "lower_80"
                        ][::-1]
                    ),
                    fill="toself",
                    line={
                        "color":
                            "rgba(100, 100, 100, 0.20)"
                    },
                    fillcolor=(
                        "rgba(100, 100, 100, 0.15)"
                    ),
                    name="80% Prediction Interval",
                )
            )

        figure.update_layout(
            height=500,
            xaxis_title="Date",
            yaxis_title="Revenue",
            hovermode="x unified",
        )

        st.plotly_chart(
            figure,
            width="stretch",
        )

        # --------------------------------------------------------------------
        # SAME-PERIOD-LAST-YEAR COMPARISON
        # --------------------------------------------------------------------

        st.subheader(
            "Same Period Last Year"
        )

        if not forecast_df.empty:

            comparison_rows = []

            for _, row in forecast_df.iterrows():

                comparison_date = (
                    row["timestamp"]
                    - pd.DateOffset(
                        years=1
                    )
                )

                matches = series_data.loc[
                    series_data[
                        "timestamp"
                    ].dt.date
                    == comparison_date.date()
                ]

                actual_last_year = None

                if not matches.empty:

                    actual_last_year = float(
                        matches.iloc[-1][
                            "target_revenue"
                        ]
                    )

                comparison_rows.append(
                    {
                        "Forecast Date":
                            row["timestamp"].date(),
                        "Forecast":
                            float(
                                row[
                                    "forecast_revenue"
                                ]
                            ),
                        "Same Date Last Year":
                            actual_last_year,
                    }
                )

            comparison_df = pd.DataFrame(
                comparison_rows
            )

            st.dataframe(
                comparison_df.style.format(
                    {
                        "Forecast":
                            "${:,.2f}",
                        "Same Date Last Year":
                            lambda value:
                                (
                                    "N/A"
                                    if pd.isna(value)
                                    else f"${value:,.2f}"
                                ),
                    }
                ),
                width="stretch",
                hide_index=True,
            )

        # --------------------------------------------------------------------
        # SHAP DRIVERS
        # --------------------------------------------------------------------

        st.subheader(
            "Forecast Drivers — TreeSHAP"
        )

        explanation = None

        if latest_timestamp is not None:

            try:

                explanation = api_get(
                    api_url,
                    "/explanation",
                    params={
                        "series_type":
                            selected_type,
                        "series_id":
                            selected_id,
                        "forecast_timestamp":
                            latest_timestamp.strftime(
                                "%Y-%m-%d"
                            ),
                    },
                )

            except Exception as exc:

                st.warning(
                    f"Unable to retrieve SHAP explanation: {exc}"
                )

        if explanation:

            driver_col1, driver_col2 = (
                st.columns(2)
            )

            with driver_col1:

                st.markdown(
                    "### Drivers pushing forecast up"
                )

                drivers_up = explanation.get(
                    "drivers_up",
                    [],
                )

                if drivers_up:

                    up_df = pd.DataFrame(
                        drivers_up
                    )

                    up_df = up_df[
                        [
                            "feature",
                            "feature_value",
                            "shap_value",
                        ]
                    ]

                    st.dataframe(
                        up_df.style.format(
                            {
                                "feature_value":
                                    "{:,.2f}",
                                "shap_value":
                                    "{:,.2f}",
                            }
                        ),
                        width="stretch",
                        hide_index=True,
                    )

            with driver_col2:

                st.markdown(
                    "### Drivers pushing forecast down"
                )

                drivers_down = explanation.get(
                    "drivers_down",
                    [],
                )

                if drivers_down:

                    down_df = pd.DataFrame(
                        drivers_down
                    )

                    down_df = down_df[
                        [
                            "feature",
                            "feature_value",
                            "shap_value",
                        ]
                    ]

                    st.dataframe(
                        down_df.style.format(
                            {
                                "feature_value":
                                    "{:,.2f}",
                                "shap_value":
                                    "{:,.2f}",
                            }
                        ),
                        width="stretch",
                        hide_index=True,
                    )

            # SHAP bar chart

            shap_rows = []

            for item in (
                explanation.get(
                    "drivers_up",
                    [],
                )
                + explanation.get(
                    "drivers_down",
                    [],
                )
            ):

                shap_rows.append(
                    {
                        "feature":
                            item[
                                "feature"
                            ],
                        "shap_value":
                            float(
                                item[
                                    "shap_value"
                                ]
                            ),
                    }
                )

            shap_df = pd.DataFrame(
                shap_rows
            )

            if not shap_df.empty:

                shap_df = (
                    shap_df
                    .sort_values(
                        "shap_value"
                    )
                )

                shap_figure = go.Figure()

                shap_figure.add_trace(
                    go.Bar(
                        x=shap_df[
                            "shap_value"
                        ],
                        y=shap_df[
                            "feature"
                        ],
                        orientation="h",
                    )
                )

                shap_figure.update_layout(
                    height=450,
                    xaxis_title=(
                        "SHAP Contribution"
                    ),
                    yaxis_title="Feature",
                )

                st.plotly_chart(
                    shap_figure,
                    width="stretch",
                )

        # --------------------------------------------------------------------
        # MONITORING
        # --------------------------------------------------------------------

        st.subheader(
            "Monitoring Status"
        )

        health_col1, health_col2, health_col3, health_col4 = (
            st.columns(4)
        )

        with health_col1:

            st.metric(
                "Overall Status",
                (
                    f"{status_emoji(monitoring_status)} "
                    f"{risk_status_label(monitoring_status)}"
                ),
            )

        with health_col2:

            st.metric(
                "Drift Alerts",
                health.get(
                    "drift_alerts",
                    "N/A",
                ),
            )

        with health_col3:

            st.metric(
                "Performance Alerts",
                health.get(
                    "performance_alerts",
                    "N/A",
                ),
            )

        with health_col4:

            st.metric(
                "Drift Warnings",
                health.get(
                    "drift_warnings",
                    "N/A",
                ),
            )

        # Series-level monitoring detail

        performance_data = (
            load_performance_data()
        )

        if performance_data is not None:

            series_performance = performance_data.loc[
                (
                    performance_data[
                        "series_type"
                    ]
                    == selected_type
                )
                & (
                    performance_data[
                        "series_id"
                    ]
                    == selected_id
                )
            ]

            if not series_performance.empty:

                st.markdown(
                    "#### Selected-series performance"
                )

                latest_performance = (
                    series_performance
                    .sort_values(
                        "recent_end"
                    )
                    .iloc[-1]
                )

                p1, p2, p3 = (
                    st.columns(3)
                )

                with p1:

                    st.metric(
                        "Recent MAPE",
                        (
                            f"{float(latest_performance['recent_mape']):.2f}%"
                        ),
                    )

                with p2:

                    st.metric(
                        "Baseline MAPE",
                        (
                            f"{float(latest_performance['baseline_mape']):.2f}%"
                        ),
                    )

                with p3:

                    st.metric(
                        "Performance Status",
                        (
                            f"{status_emoji(latest_performance.get('status'))} "
                            f"{risk_status_label(latest_performance.get('status'))}"
                        ),
                    )


# ============================================================================
# PANEL 2 — AI BUSINESS INTELLIGENCE
# ============================================================================

st.markdown("---")

st.header(
    "Panel 2 — AI Business Intelligence"
)

st.write(
    "Ask a business question in plain English. "
    "The LangGraph agent routes the request to the "
    "appropriate forecasting, SHAP, historical, RAG, "
    "and risk tools."
)

query = st.text_area(
    "Business question",
    placeholder=(
        "Example: Why is North revenue forecast risky?"
    ),
    height=100,
)

if st.button(
    "Ask the Forecasting Agent",
    type="primary",
):

    if not query.strip():

        st.warning(
            "Please enter a business question."
        )

    else:

        with st.spinner(
            "Running the forecasting agent..."
        ):

            try:

                agent_result = api_post(
                    api_url,
                    "/agent/query",
                    {
                        "query":
                            query.strip()
                    },
                )

                st.subheader(
                    "Agent Response"
                )

                st.markdown(
                    agent_result.get(
                        "synthesis",
                        "No synthesis returned.",
                    )
                )

                st.subheader(
                    "Query Interpretation"
                )

                st.json(
                    agent_result.get(
                        "query_context",
                        {},
                    )
                )

                st.subheader(
                    "Tool Plan"
                )

                st.json(
                    agent_result.get(
                        "tool_plan",
                        {},
                    )
                )

                sources = agent_result.get(
                    "sources",
                    [],
                )

                if sources:

                    st.subheader(
                        "Sources"
                    )

                    for source in sources:

                        st.markdown(
                            f"- `{source}`"
                        )

                risk = agent_result.get(
                    "risk_assessment"
                )

                if risk:

                    st.subheader(
                        "Risk Assessment"
                    )

                    risk_status = risk.get(
                        "status"
                    )

                    risk_col1, risk_col2, risk_col3 = (
                        st.columns(3)
                    )

                    with risk_col1:

                        st.metric(
                            "Status",
                            (
                                f"{status_emoji(risk_status)} "
                                f"{risk_status_label(risk_status)}"
                            ),
                        )

                    with risk_col2:

                        st.metric(
                            "Risk Score",
                            (
                                f"{risk.get('score', 'N/A')}/100"
                            ),
                        )

                    with risk_col3:

                        st.metric(
                            "Recent MAPE",
                            (
                                f"{risk.get('recent_mape', 0):.2f}%"
                                if risk.get(
                                    "recent_mape"
                                )
                                is not None
                                else "N/A"
                            ),
                        )

            except Exception as exc:

                st.error(
                    f"Agent query failed: {exc}"
                )


# ============================================================================
# LATEST WEEKLY REPORT
# ============================================================================

st.subheader(
    "Latest Automated Weekly Report"
)

try:

    latest_report = api_get(
        api_url,
        "/reports/latest",
    )

    st.caption(
        f"Report: {latest_report.get('filename', 'N/A')}"
    )

    report_content = latest_report.get(
        "content",
        "",
    )

    with st.expander(
        "View latest report",
        expanded=True,
    ):

        st.markdown(
            report_content
        )

    pdf_bytes = make_pdf(
        report_content
    )

    st.download_button(
        label="Download Weekly Report as PDF",
        data=pdf_bytes,
        file_name="weekly_business_intelligence_report.pdf",
        mime="application/pdf",
    )

except Exception as exc:

    st.warning(
        f"Latest report unavailable: {exc}"
    )