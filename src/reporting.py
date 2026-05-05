from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_datasets(clean_df: pd.DataFrame, featured_df: pd.DataFrame, clean_path: Path, featured_path: Path) -> None:
    clean_df.to_csv(clean_path, index=False)
    featured_df.to_csv(featured_path, index=False)


def export_excel_report(summary_stats: pd.DataFrame, correlation_df: pd.DataFrame, regional_summary_df: pd.DataFrame, output_path: Path) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_stats.to_excel(writer, sheet_name="summary_statistics")
        correlation_df.to_excel(writer, sheet_name="correlation_matrix")
        regional_summary_df.to_excel(writer, sheet_name="regional_summary", index=False)


def export_insights_text(insights: list[str], output_path: Path) -> None:
    output_path.write_text("\n".join(insights), encoding="utf-8")


def export_forecast_metrics(metrics_df: pd.DataFrame, output_path: Path) -> None:
    metrics_df.to_csv(output_path, index=False)
