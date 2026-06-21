from __future__ import annotations

from .config import (
    CHARTS_DIR,
    CLEAN_DATA_PATH,
    FEATURED_DATA_PATH,
    FORECAST_METRICS_PATH,
    REPORTS_DIR,
    ensure_directories,
    load_environment,
)
from .data_loader import initialize_database, load_csv_data
from .eda import build_eda_report, create_visualizations
from .feature_engineering import create_features
from .forecasting import (
    evaluate_forecasts,
    plot_forecast,
    plot_forecast_backtest,
    prepare_daily_revenue,
    forecast_future,
)
from .preprocessing import preprocess_data
from .reporting import export_datasets, export_excel_report, export_forecast_metrics, export_insights_text
from .sql_integration import extract_sales_from_sql, regional_channel_summary


def format_insights(featured_df, eda_report, regional_summary_df, forecast_df, forecast_metrics_df) -> list[str]:
    total_revenue = featured_df["net_revenue"].sum()
    total_profit = featured_df["profit"].sum()
    average_margin = featured_df["profit_margin_pct"].mean()
    top_category = featured_df.groupby("category")["net_revenue"].sum().idxmax()
    top_region = eda_report["top_region_by_profit"]
    best_channel = regional_summary_df.groupby("sales_channel")["net_revenue"].sum().idxmax()
    forecast_average = forecast_df["forecast_revenue"].mean()
    best_model = forecast_metrics_df.iloc[0]

    return [
        "Key Business Insights",
        f"Total net revenue: {total_revenue:,.2f}",
        f"Total profit: {total_profit:,.2f}",
        f"Average profit margin: {average_margin:.2f}%",
        f"Top revenue category: {top_category}",
        f"Top profit region: {top_region}",
        f"Best performing sales channel: {best_channel}",
        f"Highest revenue month: {eda_report['best_month']}",
        f"Lowest revenue month: {eda_report['worst_month']}",
        f"Best forecast model on holdout window: {best_model['model']} (RMSE: {best_model['rmse']:,.2f})",
        f"Average forecasted daily revenue for the next 30 days: {forecast_average:,.2f}",
    ]


def run_pipeline() -> dict:
    load_environment()
    ensure_directories()

    raw_df = load_csv_data()
    initialize_database(raw_df)

    sql_df = extract_sales_from_sql()
    clean_df, preprocessing_report = preprocess_data(sql_df)
    featured_df = create_features(clean_df)

    eda_report = build_eda_report(featured_df)
    chart_paths = create_visualizations(featured_df, CHARTS_DIR)

    daily_revenue = prepare_daily_revenue(featured_df)
    forecast_evaluation_df, forecast_metrics_df = evaluate_forecasts(daily_revenue, test_days=30)
    forecast_df = forecast_future(daily_revenue, future_days=30)
    forecast_chart = plot_forecast(daily_revenue, forecast_df, CHARTS_DIR)
    forecast_backtest_chart = plot_forecast_backtest(forecast_evaluation_df, CHARTS_DIR)

    regional_summary_df = regional_channel_summary()

    export_datasets(clean_df, featured_df, CLEAN_DATA_PATH, FEATURED_DATA_PATH)
    export_excel_report(
        eda_report["summary_statistics"],
        eda_report["correlation_matrix"],
        regional_summary_df,
        REPORTS_DIR / "retail_dashboard_report.xlsx",
    )
    export_forecast_metrics(forecast_metrics_df, FORECAST_METRICS_PATH)

    insights = format_insights(featured_df, eda_report, regional_summary_df, forecast_df, forecast_metrics_df)
    export_insights_text(insights, REPORTS_DIR / "business_insights.txt")

    return {
        "preprocessing_report": preprocessing_report,
        "eda_report": eda_report,
        "chart_paths": chart_paths,
        "forecast_chart": forecast_chart,
        "forecast_backtest_chart": forecast_backtest_chart,
        "forecast_metrics": forecast_metrics_df,
        "forecast_evaluation": forecast_evaluation_df,
        "insights": insights,
    }


if __name__ == "__main__":
    results = run_pipeline()
    for line in results["insights"]:
        print(line)
