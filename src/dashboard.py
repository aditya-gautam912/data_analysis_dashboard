from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.config import FEATURED_DATA_PATH
    from src.main import run_pipeline
else:
    from .config import FEATURED_DATA_PATH
    from .main import run_pipeline


st.set_page_config(page_title="Retail Analytics Dashboard", layout="wide")
sns.set_theme(style="whitegrid")


@st.cache_data
def load_dashboard_data() -> pd.DataFrame:
    if not FEATURED_DATA_PATH.exists():
        run_pipeline()
    df = pd.read_csv(FEATURED_DATA_PATH, parse_dates=["order_date", "ship_date"])
    return df


@st.cache_data
def load_pipeline_results() -> dict:
    return run_pipeline()


def metric_card(label: str, value: str) -> None:
    st.metric(label=label, value=value)


def safe_pct_change(current_value: float, previous_value: float) -> str:
    if previous_value == 0:
        return "n/a"
    return f"{((current_value - previous_value) / previous_value) * 100:.2f}%"


def main() -> None:
    st.title("Data Analysis & Visualization Dashboard")
    st.caption("Retail sales, profitability, trend analysis, and forecasting")

    df = load_dashboard_data()
    comparison_df = load_pipeline_results()["forecast_metrics"]

    min_date = df["order_date"].min().date()
    max_date = df["order_date"].max().date()

    st.sidebar.header("Filters")
    selected_regions = st.sidebar.multiselect("Region", sorted(df["region"].unique()), default=sorted(df["region"].unique()))
    selected_categories = st.sidebar.multiselect("Category", sorted(df["category"].unique()), default=sorted(df["category"].unique()))
    selected_segments = st.sidebar.multiselect(
        "Customer Segment",
        sorted(df["customer_segment"].unique()),
        default=sorted(df["customer_segment"].unique()),
    )
    selected_channels = st.sidebar.multiselect("Sales Channel", sorted(df["sales_channel"].unique()), default=sorted(df["sales_channel"].unique()))
    date_granularity = st.sidebar.radio("Trend Granularity", ["Daily", "Monthly"], index=1)
    selected_metric = st.sidebar.selectbox("Comparison Metric", ["net_revenue", "profit", "units_sold"], index=0)
    selected_dates = st.sidebar.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date = min_date
        end_date = max_date
    filtered_df = df[
        (df["region"].isin(selected_regions))
        & (df["category"].isin(selected_categories))
        & (df["customer_segment"].isin(selected_segments))
        & (df["sales_channel"].isin(selected_channels))
        & (df["order_date"].dt.date >= start_date)
        & (df["order_date"].dt.date <= end_date)
    ].copy()

    if filtered_df.empty:
        st.warning("No rows match the selected filters.")
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

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Revenue", f"{total_revenue:,.0f}", delta=safe_pct_change(total_revenue, previous_df["net_revenue"].sum()))
    with col2:
        st.metric("Profit", f"{total_profit:,.0f}", delta=safe_pct_change(total_profit, previous_df["profit"].sum()))
    with col3:
        st.metric(
            "Avg Margin %",
            f"{average_margin:.2f}",
            delta=safe_pct_change(average_margin, previous_df["profit_margin_pct"].mean() if not previous_df.empty else 0),
        )
    with col4:
        st.metric("Orders", f"{total_orders:,}", delta=safe_pct_change(total_orders, len(previous_df)))

    st.subheader("Trend Analysis")
    if date_granularity == "Daily":
        trend_df = filtered_df.groupby("order_date", as_index=False)[selected_metric].sum().sort_values("order_date")
        trend_x = "order_date"
        trend_title = f"Daily {selected_metric.replace('_', ' ').title()}"
    else:
        trend_df = filtered_df.groupby("year_month", as_index=False)[selected_metric].sum().sort_values("year_month")
        trend_x = "year_month"
        trend_title = f"Monthly {selected_metric.replace('_', ' ').title()}"

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.lineplot(data=trend_df, x=trend_x, y=selected_metric, ax=ax, linewidth=2, marker="o")
    ax.set_title(trend_title)
    ax.set_xlabel("Date")
    ax.set_ylabel(selected_metric.replace("_", " ").title())
    if date_granularity == "Monthly":
        ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)
    plt.close(fig)

    overview_tab, forecast_tab, drilldown_tab = st.tabs(["Overview", "Forecast", "Drilldown"])

    with overview_tab:
        left, right = st.columns(2)

        with left:
            st.subheader("Revenue by Category")
            category_df = filtered_df.groupby("category", as_index=False)["net_revenue"].sum().sort_values("net_revenue", ascending=False)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=category_df, x="category", y="net_revenue", ax=ax)
            ax.set_xlabel("Category")
            ax.set_ylabel("Revenue")
            st.pyplot(fig)
            plt.close(fig)

        with right:
            st.subheader("Profit Distribution")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(filtered_df["profit"], kde=True, bins=30, ax=ax)
            ax.set_xlabel("Profit")
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Correlation Heatmap")
        correlation_columns = [
            "units_sold",
            "unit_price",
            "unit_cost",
            "discount_pct",
            "marketing_spend",
            "inventory_days",
            "customer_rating",
            "net_revenue",
            "profit",
            "ship_delay_days",
        ]
        corr = filtered_df[correlation_columns].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, cmap="YlGnBu", annot=True, fmt=".2f", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    with forecast_tab:
        st.subheader("Forecast Model Evaluation")
        st.dataframe(comparison_df, use_container_width=True)

        forecast_base = (
            filtered_df.groupby("order_date", as_index=False)["net_revenue"].sum().sort_values("order_date")
        )
        forecast_base["moving_average_14"] = forecast_base["net_revenue"].rolling(window=14, min_periods=1).mean()
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.lineplot(data=forecast_base, x="order_date", y="net_revenue", ax=ax, label="Actual Revenue")
        sns.lineplot(data=forecast_base, x="order_date", y="moving_average_14", ax=ax, label="14-Day Moving Average")
        ax.set_title("Filtered Revenue Trend With Smoothing")
        st.pyplot(fig)
        plt.close(fig)

    with drilldown_tab:
        st.subheader("Regional and Product Drilldown")
        drilldown_metric = st.selectbox("Drilldown Metric", ["net_revenue", "profit", "units_sold"], index=0)

        left, right = st.columns(2)
        with left:
            region_table = (
                filtered_df.groupby("region", as_index=False)[drilldown_metric]
                .sum()
                .sort_values(drilldown_metric, ascending=False)
            )
            st.dataframe(region_table, use_container_width=True)

        with right:
            product_table = (
                filtered_df.groupby(["category", "sub_category"], as_index=False)[drilldown_metric]
                .sum()
                .sort_values(drilldown_metric, ascending=False)
                .head(15)
            )
            st.dataframe(product_table, use_container_width=True)

        st.subheader("Detailed Data Preview")
        preview_columns = [
            "order_id",
            "order_date",
            "region",
            "customer_segment",
            "category",
            "sub_category",
            "sales_channel",
            "units_sold",
            "net_revenue",
            "profit",
            "profit_margin_pct",
        ]
        preview_df = filtered_df[preview_columns].sort_values("order_date", ascending=False)
        st.download_button(
            "Download Filtered Data as CSV",
            data=preview_df.to_csv(index=False).encode("utf-8"),
            file_name="filtered_retail_data.csv",
            mime="text/csv",
        )
        st.dataframe(preview_df, use_container_width=True)


if __name__ == "__main__":
    main()
