from __future__ import annotations

import numpy as np
import pandas as pd


def create_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()

    df["gross_sales"] = df["units_sold"] * df["unit_price"]
    df["discount_amount"] = df["gross_sales"] * df["discount_pct"]
    df["net_revenue"] = df["gross_sales"] - df["discount_amount"]
    df["total_cost"] = df["units_sold"] * df["unit_cost"]
    df["profit"] = df["net_revenue"] - df["total_cost"] - df["marketing_spend"]
    df["profit_margin_pct"] = np.where(df["net_revenue"] > 0, (df["profit"] / df["net_revenue"]) * 100, 0)
    df["ship_delay_days"] = (df["ship_date"] - df["order_date"]).dt.days
    df["is_weekend_order"] = df["order_date"].dt.dayofweek >= 5
    df["order_year"] = df["order_date"].dt.year
    df["order_month"] = df["order_date"].dt.month
    df["order_month_name"] = df["order_date"].dt.strftime("%b")
    df["year_month"] = df["order_date"].dt.to_period("M").astype(str)
    df["quarter"] = df["order_date"].dt.to_period("Q").astype(str)
    df["average_selling_price"] = np.where(df["units_sold"] > 0, df["net_revenue"] / df["units_sold"], 0)
    df["return_flag"] = df["returned"].map({1: "Returned", 0: "Not Returned"})
    df["high_discount_flag"] = np.where(df["discount_pct"] >= 0.15, "High", "Normal")

    return df
