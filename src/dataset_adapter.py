from __future__ import annotations

import numpy as np
import pandas as pd


def _clean_column_names(columns: pd.Index) -> list[str]:
    return [
        str(column).strip().lower().replace("-", "_").replace(" ", "_")
        for column in columns
    ]


def _normalize_current_schema(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe.copy()


def _normalize_superstore_schema(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    df.columns = _clean_column_names(df.columns)

    required_columns = {
        "order_id",
        "order_date",
        "ship_date",
        "region",
        "state",
        "city",
        "segment",
        "category",
        "sub_category",
        "product_name",
        "sales",
        "quantity",
        "discount",
        "profit",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for Superstore-style CSV: {sorted(missing)}")

    safe_quantity = df["quantity"].replace(0, 1)
    net_unit_price = df["sales"] / safe_quantity
    estimated_unit_cost = (df["sales"] - df["profit"]) / safe_quantity

    normalized = pd.DataFrame(
        {
            "order_id": df["order_id"].astype(str),
            "order_date": df["order_date"],
            "ship_date": df["ship_date"],
            "region": df["region"],
            "state": df["state"],
            "city": df["city"],
            "sales_channel": "External CSV",
            "customer_segment": df["segment"],
            "category": df["category"],
            "sub_category": df["sub_category"],
            "product_name": df["product_name"],
            "units_sold": df["quantity"],
            "unit_price": net_unit_price.round(2),
            "unit_cost": estimated_unit_cost.clip(lower=0.01).round(2),
            "discount_pct": df["discount"].clip(lower=0, upper=0.5),
            "marketing_spend": np.maximum(df["sales"] * 0.03, 5).round(2),
            "inventory_days": 14,
            "returned": 0,
            "customer_rating": 4.0,
        }
    )
    return normalized


def normalize_retail_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    column_names = set(dataframe.columns)
    current_schema = {
        "order_id",
        "order_date",
        "ship_date",
        "region",
        "state",
        "city",
        "sales_channel",
        "customer_segment",
        "category",
        "sub_category",
        "product_name",
        "units_sold",
        "unit_price",
        "unit_cost",
        "discount_pct",
        "marketing_spend",
        "inventory_days",
        "returned",
        "customer_rating",
    }
    if current_schema.issubset(column_names):
        return _normalize_current_schema(dataframe)

    cleaned_columns = set(_clean_column_names(pd.Index(dataframe.columns)))
    superstore_schema = {
        "order_id",
        "order_date",
        "ship_date",
        "region",
        "state",
        "city",
        "segment",
        "category",
        "sub_category",
        "product_name",
        "sales",
        "quantity",
        "discount",
        "profit",
    }
    if superstore_schema.issubset(cleaned_columns):
        return _normalize_superstore_schema(dataframe)

    raise ValueError(
        "Unsupported dataset schema. Provide either the project's retail schema or a Superstore-style CSV."
    )
