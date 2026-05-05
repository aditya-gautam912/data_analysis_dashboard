from __future__ import annotations

import numpy as np
import pandas as pd


def preprocess_data(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = dataframe.copy()
    report = {}
    numeric_fill_defaults = {
        "units_sold": 1,
        "unit_price": 0.01,
        "unit_cost": 0.01,
        "discount_pct": 0.0,
        "marketing_spend": 0.0,
        "inventory_days": 1,
        "customer_rating": 4.0,
    }

    report["initial_shape"] = df.shape
    report["missing_values_before"] = df.isna().sum().to_dict()

    df = df.drop_duplicates()
    report["duplicates_removed"] = report["initial_shape"][0] - len(df)

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["ship_date"] = pd.to_datetime(df["ship_date"], errors="coerce")
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    df["unit_cost"] = pd.to_numeric(df["unit_cost"], errors="coerce")
    df["discount_pct"] = pd.to_numeric(df["discount_pct"], errors="coerce")
    df["marketing_spend"] = pd.to_numeric(df["marketing_spend"], errors="coerce")
    df["inventory_days"] = pd.to_numeric(df["inventory_days"], errors="coerce")
    df["returned"] = pd.to_numeric(df["returned"], errors="coerce").fillna(0).astype(int)
    df["customer_rating"] = pd.to_numeric(df["customer_rating"], errors="coerce")

    numeric_columns = [
        "units_sold",
        "unit_price",
        "unit_cost",
        "discount_pct",
        "marketing_spend",
        "inventory_days",
        "customer_rating",
    ]
    categorical_columns = [
        "region",
        "state",
        "city",
        "sales_channel",
        "customer_segment",
        "category",
        "sub_category",
        "product_name",
    ]

    for column in numeric_columns:
        if df[column].notna().any():
            fill_value = df[column].median()
        else:
            fill_value = numeric_fill_defaults[column]
        df[column] = df[column].fillna(fill_value)

    for column in categorical_columns:
        mode = df[column].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "Unknown"
        df[column] = df[column].fillna(fill_value)

    df = df.dropna(subset=["order_date", "ship_date"])

    df["discount_pct"] = df["discount_pct"].clip(0, 0.5)
    df["customer_rating"] = df["customer_rating"].clip(1, 5)
    df["inventory_days"] = df["inventory_days"].clip(lower=1)
    df["units_sold"] = df["units_sold"].clip(lower=1)
    df["unit_cost"] = df["unit_cost"].clip(lower=0.01)
    df["unit_price"] = df["unit_price"].clip(lower=0.01)

    standardized_columns = ["unit_price", "unit_cost", "marketing_spend", "inventory_days"]
    for column in standardized_columns:
        std = df[column].std(ddof=0)
        if std == 0:
            df[f"{column}_zscore"] = 0.0
        else:
            df[f"{column}_zscore"] = (df[column] - df[column].mean()) / std

    report["final_shape"] = df.shape
    report["missing_values_after"] = df.isna().sum().to_dict()
    report["date_range"] = {
        "start": df["order_date"].min().strftime("%Y-%m-%d"),
        "end": df["order_date"].max().strftime("%Y-%m-%d"),
    }

    return df.sort_values("order_date").reset_index(drop=True), report


def detect_outliers_iqr(dataframe: pd.DataFrame, columns: list[str]) -> dict:
    outlier_summary = {}
    for column in columns:
        q1 = dataframe[column].quantile(0.25)
        q3 = dataframe[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (dataframe[column] < lower) | (dataframe[column] > upper)
        outlier_summary[column] = int(mask.sum())
    return outlier_summary
