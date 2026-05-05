import pandas as pd

from src.feature_engineering import create_features


def test_create_features_adds_profit_and_margin_columns():
    dataframe = pd.DataFrame(
        {
            "order_date": pd.to_datetime(["2024-01-01"]),
            "ship_date": pd.to_datetime(["2024-01-04"]),
            "units_sold": [2],
            "unit_price": [500.0],
            "unit_cost": [300.0],
            "discount_pct": [0.1],
            "marketing_spend": [50.0],
            "returned": [0],
        }
    )

    featured_df = create_features(dataframe)

    assert featured_df["gross_sales"].iloc[0] == 1000.0
    assert featured_df["discount_amount"].iloc[0] == 100.0
    assert featured_df["net_revenue"].iloc[0] == 900.0
    assert featured_df["total_cost"].iloc[0] == 600.0
    assert featured_df["profit"].iloc[0] == 250.0
    assert round(featured_df["profit_margin_pct"].iloc[0], 2) == 27.78
    assert featured_df["ship_delay_days"].iloc[0] == 3
