from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .preprocessing import detect_outliers_iqr


sns.set_theme(style="whitegrid", palette="deep")


def generate_summary_statistics(dataframe: pd.DataFrame) -> pd.DataFrame:
    columns = ["units_sold", "unit_price", "net_revenue", "profit", "marketing_spend", "customer_rating"]
    return dataframe[columns].agg(["mean", "median", "std", "min", "max"]).round(2)


def correlation_matrix(dataframe: pd.DataFrame) -> pd.DataFrame:
    columns = [
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
    return dataframe[columns].corr(numeric_only=True).round(2)


def create_visualizations(dataframe: pd.DataFrame, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = {}

    revenue_by_category = dataframe.groupby("category", as_index=False)["net_revenue"].sum().sort_values("net_revenue", ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=revenue_by_category, x="category", y="net_revenue")
    plt.title("Revenue by Category")
    plt.xlabel("Category")
    plt.ylabel("Net Revenue")
    plt.xticks(rotation=15)
    category_path = output_dir / "bar_revenue_by_category.png"
    plt.tight_layout()
    plt.savefig(category_path, dpi=200)
    plt.close()
    saved_paths["bar_revenue_by_category"] = str(category_path)

    daily_sales = dataframe.groupby("order_date", as_index=False)["net_revenue"].sum().sort_values("order_date")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=daily_sales, x="order_date", y="net_revenue", linewidth=1.8)
    plt.title("Daily Revenue Trend")
    plt.xlabel("Order Date")
    plt.ylabel("Net Revenue")
    trend_path = output_dir / "line_daily_revenue_trend.png"
    plt.tight_layout()
    plt.savefig(trend_path, dpi=200)
    plt.close()
    saved_paths["line_daily_revenue_trend"] = str(trend_path)

    corr = correlation_matrix(dataframe)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f", square=True)
    plt.title("Correlation Heatmap")
    heatmap_path = output_dir / "heatmap_correlation.png"
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=200)
    plt.close()
    saved_paths["heatmap_correlation"] = str(heatmap_path)

    plt.figure(figsize=(10, 6))
    sns.histplot(data=dataframe, x="profit", kde=True, bins=40)
    plt.title("Profit Distribution")
    plt.xlabel("Profit")
    distribution_path = output_dir / "distribution_profit.png"
    plt.tight_layout()
    plt.savefig(distribution_path, dpi=200)
    plt.close()
    saved_paths["distribution_profit"] = str(distribution_path)

    monthly_region_trend = (
        dataframe.groupby(["year_month", "region"], as_index=False)["net_revenue"].sum().sort_values("year_month")
    )
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=monthly_region_trend, x="year_month", y="net_revenue", hue="region", marker="o")
    plt.title("Monthly Revenue Trend by Region")
    plt.xlabel("Year-Month")
    plt.ylabel("Net Revenue")
    plt.xticks(rotation=45)
    region_trend_path = output_dir / "trend_monthly_region_revenue.png"
    plt.tight_layout()
    plt.savefig(region_trend_path, dpi=200)
    plt.close()
    saved_paths["trend_monthly_region_revenue"] = str(region_trend_path)

    return saved_paths


def build_eda_report(dataframe: pd.DataFrame) -> dict:
    outliers = detect_outliers_iqr(dataframe, ["units_sold", "unit_price", "net_revenue", "profit"])
    monthly_trend = dataframe.groupby("year_month")["net_revenue"].sum()
    best_month = monthly_trend.idxmax()
    worst_month = monthly_trend.idxmin()

    region_profit = dataframe.groupby("region")["profit"].sum().sort_values(ascending=False)
    top_region = region_profit.index[0]

    return {
        "summary_statistics": generate_summary_statistics(dataframe),
        "correlation_matrix": correlation_matrix(dataframe),
        "outliers": outliers,
        "best_month": best_month,
        "worst_month": worst_month,
        "top_region_by_profit": top_region,
    }
