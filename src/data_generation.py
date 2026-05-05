from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample_retail_dataset(output_path: Path, random_seed: int = 42, rows: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)

    regions = {
        "North": [("Delhi", "New Delhi"), ("Punjab", "Ludhiana"), ("Haryana", "Gurugram")],
        "South": [("Karnataka", "Bengaluru"), ("Tamil Nadu", "Chennai"), ("Telangana", "Hyderabad")],
        "West": [("Maharashtra", "Mumbai"), ("Gujarat", "Ahmedabad"), ("Rajasthan", "Jaipur")],
        "East": [("West Bengal", "Kolkata"), ("Odisha", "Bhubaneswar"), ("Bihar", "Patna")],
    }
    category_map = {
        "Technology": [("Laptops", "Business Laptop"), ("Phones", "5G Smartphone"), ("Accessories", "Wireless Headset")],
        "Furniture": [("Chairs", "Ergonomic Chair"), ("Tables", "Standing Desk"), ("Storage", "Office Cabinet")],
        "Office Supplies": [("Paper", "Printer Paper"), ("Binders", "Ring Binder"), ("Appliances", "Label Printer")],
    }
    channels = ["Online", "Retail Store", "Distributor"]
    segments = ["Consumer", "Corporate", "Home Office", "Small Business"]

    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    sampled_dates = rng.choice(dates, size=rows, replace=True)

    records = []
    for index in range(rows):
        region = rng.choice(list(regions.keys()), p=[0.28, 0.24, 0.27, 0.21])
        state, city = regions[region][rng.integers(0, len(regions[region]))]

        category = rng.choice(list(category_map.keys()), p=[0.42, 0.24, 0.34])
        sub_category, product_name = category_map[category][rng.integers(0, len(category_map[category]))]

        units_sold = int(max(1, rng.poisson(4)))
        base_price = {
            "Technology": rng.uniform(120, 1600),
            "Furniture": rng.uniform(80, 900),
            "Office Supplies": rng.uniform(10, 240),
        }[category]
        unit_price = round(float(base_price), 2)
        unit_cost = round(float(unit_price * rng.uniform(0.45, 0.82)), 2)
        discount_pct = round(float(rng.choice([0.0, 0.05, 0.1, 0.15, 0.2], p=[0.25, 0.22, 0.25, 0.18, 0.10])), 2)
        marketing_spend = round(float(rng.uniform(10, 120)), 2)
        inventory_days = int(max(1, rng.normal(18, 6)))
        returned = int(rng.choice([0, 1], p=[0.91, 0.09]))
        customer_rating = round(float(np.clip(rng.normal(4.1, 0.55), 1, 5)), 1)

        order_date = pd.Timestamp(sampled_dates[index])
        ship_delay = int(max(1, rng.normal(4, 1.5)))
        ship_date = order_date + pd.Timedelta(days=ship_delay)

        records.append(
            {
                "order_id": f"ORD-{100000 + index}",
                "order_date": order_date.strftime("%Y-%m-%d"),
                "ship_date": ship_date.strftime("%Y-%m-%d"),
                "region": region,
                "state": state,
                "city": city,
                "sales_channel": rng.choice(channels, p=[0.53, 0.28, 0.19]),
                "customer_segment": rng.choice(segments, p=[0.48, 0.24, 0.16, 0.12]),
                "category": category,
                "sub_category": sub_category,
                "product_name": product_name,
                "units_sold": units_sold,
                "unit_price": unit_price,
                "unit_cost": unit_cost,
                "discount_pct": discount_pct,
                "marketing_spend": marketing_spend,
                "inventory_days": inventory_days,
                "returned": returned,
                "customer_rating": customer_rating,
            }
        )

    dataframe = pd.DataFrame(records)

    missing_indices = rng.choice(dataframe.index, size=120, replace=False)
    dataframe.loc[missing_indices[:35], "customer_rating"] = np.nan
    dataframe.loc[missing_indices[35:70], "marketing_spend"] = np.nan
    dataframe.loc[missing_indices[70:95], "sales_channel"] = np.nan
    dataframe.loc[missing_indices[95:], "discount_pct"] = np.nan

    duplicate_rows = dataframe.sample(35, random_state=random_seed)
    dataframe = pd.concat([dataframe, duplicate_rows], ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    return dataframe
