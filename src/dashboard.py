from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.config import FEATURED_DATA_PATH, FORECAST_METRICS_PATH
    from src.main import run_pipeline
else:
    from .config import FEATURED_DATA_PATH, FORECAST_METRICS_PATH
    from .main import run_pipeline


st.set_page_config(
    page_title="Retail Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.03em;
    }

    .main-header {
        background: linear-gradient(135deg, #6c63ff, #48c6ef);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.25rem !important;
    }

    .main-subtitle {
        color: rgba(255,255,255,0.5);
        font-size: 0.9rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #6c63ff, #48c6ef);
        opacity: 0.6;
    }

    .metric-card:hover {
        background: rgba(255,255,255,0.07);
        border-color: rgba(108, 99, 255, 0.3);
        transform: translateY(-2px);
    }

    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: rgba(255,255,255,0.5);
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #fff;
        letter-spacing: -0.02em;
    }

    .metric-delta {
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 0.25rem;
    }

    .metric-delta.positive { color: #4ade80; }
    .metric-delta.negative { color: #f87171; }

    div[data-testid="stSidebar"] {
        background: rgba(15, 15, 26, 0.95);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    div[data-testid="stSidebar"] .sidebar-content {
        padding-top: 1.5rem;
    }

    .sidebar-header {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: rgba(255,255,255,0.3);
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }

    .stMultiSelect div[data-baseweb="select"] > div,
    .stSelectbox div[data-baseweb="select"] > div,
    .stRadio div[role="radiogroup"] {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
    }

    div.stDateInput input {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
        color: #fff !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.85rem;
        color: rgba(255,255,255,0.4);
        padding: 0.75rem 1.25rem;
        border-radius: 10px 10px 0 0;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        color: #6c63ff !important;
        background: rgba(108, 99, 255, 0.08) !important;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        overflow: hidden;
    }

    .stDownloadButton button {
        background: linear-gradient(135deg, #6c63ff, #48c6ef) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
        transition: all 0.3s ease !important;
    }

    .stDownloadButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(108, 99, 255, 0.4) !important;
    }

    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(108, 99, 255, 0.3), transparent);
        margin: 2rem 0;
    }

    footer { display: none; }
</style>
"""


@st.cache_data
def load_dashboard_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not FEATURED_DATA_PATH.exists():
        run_pipeline()
    df = pd.read_csv(FEATURED_DATA_PATH, parse_dates=["order_date", "ship_date"])
    forecast_metrics = pd.read_csv(FORECAST_METRICS_PATH) if FORECAST_METRICS_PATH.exists() else pd.DataFrame()
    return df, forecast_metrics


def safe_pct_change(current_value: float, previous_value: float) -> tuple[str, bool]:
    if previous_value == 0:
        return "n/a", True
    pct = ((current_value - previous_value) / previous_value) * 100
    return f"{pct:+.2f}%", pct >= 0


def render_metric_card(label: str, value: str, delta: str, positive: bool) -> None:
    delta_class = "positive" if positive else "negative"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta {delta_class}">{delta} vs prior period</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, sans-serif", "color": "rgba(255,255,255,0.7)"},
        "title": {"font": {"size": 16, "color": "#fff", "weight": 700}},
        "xaxis": {
            "gridcolor": "rgba(255,255,255,0.04)",
            "zerolinecolor": "rgba(255,255,255,0.06)",
            "tickfont": {"size": 11},
        },
        "yaxis": {
            "gridcolor": "rgba(255,255,255,0.04)",
            "zerolinecolor": "rgba(255,255,255,0.06)",
            "tickfont": {"size": 11},
        },
        "hoverlabel": {
            "bgcolor": "#1a1a2e",
            "font": {"color": "#fff", "size": 12},
            "bordercolor": "rgba(108, 99, 255, 0.3)",
        },
        "margin": {"l": 10, "r": 10, "t": 40, "b": 10},
        "legend": {
            "font": {"size": 11, "color": "rgba(255,255,255,0.6)"},
            "bgcolor": "rgba(0,0,0,0)",
        },
    }
}


def create_trend_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode="lines+markers",
            name=y_col.replace("_", " ").title(),
            line=dict(color="#6c63ff", width=3),
            marker=dict(color="#6c63ff", size=6, symbol="circle", line=dict(width=1, color="#fff")),
            fill="tozeroy",
            fillcolor="rgba(108, 99, 255, 0.06)",
        )
    )
    fig.update_layout(
        title=title,
        **PLOTLY_TEMPLATE["layout"],
        hovermode="x unified",
    )
    return fig


def create_category_chart(df: pd.DataFrame) -> go.Figure:
    colors = ["#6c63ff", "#48c6ef", "#fbbf24"]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["category"],
            y=df["net_revenue"],
            marker=dict(color=colors[: len(df)], line=dict(width=0)),
            text=df["net_revenue"].apply(lambda x: f"${x:,.0f}"),
            textposition="outside",
            textfont=dict(size=11, color="rgba(255,255,255,0.7)"),
        )
    )
    fig.update_layout(
        title="Revenue by Category",
        xaxis_title=None,
        yaxis_title="Revenue",
        **PLOTLY_TEMPLATE["layout"],
        yaxis={"gridcolor": "rgba(255,255,255,0.04)", "tickfont": {"size": 11}},
    )
    return fig


def create_profit_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=df["profit"],
            nbinsx=30,
            marker=dict(color="#6c63ff", line=dict(width=0)),
            hovertemplate="Profit: $%{x:,.2f}<br>Count: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Profit Distribution",
        xaxis_title="Profit",
        yaxis_title="Frequency",
        bargap=0.05,
        **PLOTLY_TEMPLATE["layout"],
    )
    return fig


def create_heatmap(df: pd.DataFrame, columns: list[str]) -> go.Figure:
    corr = df[columns].corr(numeric_only=True)
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="Viridis",
            zmin=-1,
            zmax=1,
            text=corr.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 10, "color": "#fff"},
            hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Correlation Heatmap",
        xaxis=dict(tickangle=45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
        **PLOTLY_TEMPLATE["layout"],
        margin={"l": 10, "r": 10, "t": 40, "b": 120},
    )
    return fig


def create_forecast_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["order_date"],
            y=df["net_revenue"],
            mode="lines",
            name="Actual Revenue",
            line=dict(color="#6c63ff", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["order_date"],
            y=df["moving_average_14"],
            mode="lines",
            name="14-Day Moving Avg",
            line=dict(color="#fbbf24", width=2, dash="dot"),
        )
    )
    fig.update_layout(
        title="Revenue Trend with Smoothing",
        hovermode="x unified",
        **PLOTLY_TEMPLATE["layout"],
    )
    return fig


def style_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    styled = df.copy()
    for col in styled.select_dtypes(include=[np.number]).columns:
        if styled[col].dtype in ("float64", "float32"):
            styled[col] = styled[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
        elif styled[col].dtype in ("int64", "int32"):
            styled[col] = styled[col].apply(lambda x: f"{x:,}" if pd.notna(x) else "")
    return styled


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">Data Analysis & Visualization Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Retail sales, profitability, trend analysis, and forecasting</p>', unsafe_allow_html=True)

    df, comparison_df = load_dashboard_data()

    min_date = df["order_date"].min().date()
    max_date = df["order_date"].max().date()

    with st.sidebar:
        st.markdown('<div class="sidebar-header">Filters</div>', unsafe_allow_html=True)
        selected_regions = st.multiselect("Region", sorted(df["region"].unique()), default=sorted(df["region"].unique()))
        selected_categories = st.multiselect("Category", sorted(df["category"].unique()), default=sorted(df["category"].unique()))
        selected_segments = st.multiselect(
            "Customer Segment",
            sorted(df["customer_segment"].unique()),
            default=sorted(df["customer_segment"].unique()),
        )
        selected_channels = st.multiselect("Sales Channel", sorted(df["sales_channel"].unique()), default=sorted(df["sales_channel"].unique()))

        st.markdown('<div class="sidebar-header" style="margin-top:1.5rem">Settings</div>', unsafe_allow_html=True)
        date_granularity = st.radio("Trend Granularity", ["Daily", "Monthly"], index=1, horizontal=True)
        selected_metric = st.selectbox("Comparison Metric", ["net_revenue", "profit", "units_sold"], index=0)
        selected_dates = st.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date, end_date = min_date, max_date

    filtered_df = df[
        (df["region"].isin(selected_regions))
        & (df["category"].isin(selected_categories))
        & (df["customer_segment"].isin(selected_segments))
        & (df["sales_channel"].isin(selected_channels))
        & (df["order_date"].dt.date >= start_date)
        & (df["order_date"].dt.date <= end_date)
    ].copy()

    if filtered_df.empty:
        st.warning("No rows match the selected filters. Adjust your filter criteria.")
        return

    total_revenue = filtered_df["net_revenue"].sum()
    total_profit = filtered_df["profit"].sum()
    average_margin = filtered_df["profit_margin_pct"].mean()
    total_orders = len(filtered_df)

    comparison_start = pd.Timestamp(start_date)
    comparison_end = pd.Timestamp(end_date)
    selected_window_days = max((comparison_end - comparison_start).days + 1, 1)
    previous_end = comparison_start - pd.Timedelta(days=1)
    previous_start = previous_end - pd.Timedelta(days=selected_window_days - 1)
    previous_df = df[
        (df["order_date"] >= previous_start)
        & (df["order_date"] <= previous_end)
        & (df["region"].isin(selected_regions))
        & (df["category"].isin(selected_categories))
        & (df["customer_segment"].isin(selected_segments))
        & (df["sales_channel"].isin(selected_channels))
    ].copy()

    cols = st.columns(4)
    with cols[0]:
        delta, pos = safe_pct_change(total_revenue, previous_df["net_revenue"].sum())
        render_metric_card("Revenue", f"${total_revenue:,.0f}", delta, pos)
    with cols[1]:
        delta, pos = safe_pct_change(total_profit, previous_df["profit"].sum())
        render_metric_card("Profit", f"${total_profit:,.0f}", delta, pos)
    with cols[2]:
        prev_margin = previous_df["profit_margin_pct"].mean() if not previous_df.empty else 0
        delta, pos = safe_pct_change(average_margin, prev_margin)
        render_metric_card("Avg Margin", f"{average_margin:.2f}%", delta, pos)
    with cols[3]:
        delta, pos = safe_pct_change(total_orders, len(previous_df))
        render_metric_card("Orders", f"{total_orders:,}", delta, pos)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.subheader("Trend Analysis", anchor=False)
    if date_granularity == "Daily":
        trend_df = filtered_df.groupby("order_date", as_index=False)[selected_metric].sum().sort_values("order_date")
        trend_x = "order_date"
        trend_title = f"Daily {selected_metric.replace('_', ' ').title()}"
    else:
        trend_df = filtered_df.groupby("year_month", as_index=False)[selected_metric].sum().sort_values("year_month")
        trend_x = "year_month"
        trend_title = f"Monthly {selected_metric.replace('_', ' ').title()}"

    fig = create_trend_chart(trend_df, trend_x, selected_metric, trend_title)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    overview_tab, forecast_tab, drilldown_tab = st.tabs(["Overview", "Forecast", "Drilldown"])

    with overview_tab:
        left, right = st.columns(2)

        with left:
            category_df = filtered_df.groupby("category", as_index=False)["net_revenue"].sum().sort_values("net_revenue", ascending=False)
            fig = create_category_chart(category_df)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with right:
            fig = create_profit_chart(filtered_df)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        correlation_columns = [
            "units_sold", "unit_price", "unit_cost", "discount_pct",
            "marketing_spend", "inventory_days", "customer_rating",
            "net_revenue", "profit", "ship_delay_days",
        ]
        fig = create_heatmap(filtered_df, correlation_columns)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with forecast_tab:
        st.subheader("Forecast Model Evaluation", anchor=False)
        if not comparison_df.empty:
            st.dataframe(style_dataframe(comparison_df), use_container_width=True, hide_index=True)
        else:
            st.info("Run the pipeline with forecasting enabled to see model evaluation.")

        forecast_base = filtered_df.groupby("order_date", as_index=False)["net_revenue"].sum().sort_values("order_date")
        forecast_base["moving_average_14"] = forecast_base["net_revenue"].rolling(window=14, min_periods=1).mean()
        fig = create_forecast_chart(forecast_base)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with drilldown_tab:
        st.subheader("Regional & Product Drilldown", anchor=False)
        drilldown_metric = st.selectbox("Drilldown Metric", ["net_revenue", "profit", "units_sold"], index=0, key="drilldown_metric")

        left, right = st.columns(2)
        with left:
            region_table = (
                filtered_df.groupby("region", as_index=False)[drilldown_metric]
                .sum()
                .sort_values(drilldown_metric, ascending=False)
            )
            st.caption("By Region")
            st.dataframe(style_dataframe(region_table), use_container_width=True, hide_index=True)

        with right:
            product_table = (
                filtered_df.groupby(["category", "sub_category"], as_index=False)[drilldown_metric]
                .sum()
                .sort_values(drilldown_metric, ascending=False)
                .head(15)
            )
            st.caption("Top 15 Product Sub-Categories")
            st.dataframe(style_dataframe(product_table), use_container_width=True, hide_index=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        preview_columns = [
            "order_id", "order_date", "region", "customer_segment",
            "category", "sub_category", "sales_channel", "units_sold",
            "net_revenue", "profit", "profit_margin_pct",
        ]
        preview_df = filtered_df[preview_columns].sort_values("order_date", ascending=False)
        st.subheader("Data Preview", anchor=False)
        dl_col, _ = st.columns([1, 5])
        with dl_col:
            st.download_button(
                "Download CSV",
                data=preview_df.to_csv(index=False).encode("utf-8"),
                file_name="filtered_retail_data.csv",
                mime="text/csv",
            )
        st.dataframe(style_dataframe(preview_df), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
