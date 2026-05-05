import pandas as pd

from src.preprocessing import preprocess_data


def test_preprocess_data_removes_duplicates_and_fills_missing_values():
    dataframe = pd.DataFrame(
        [
            {
                "order_id": "ORD-1",
                "order_date": "2024-01-01",
                "ship_date": "2024-01-03",
                "region": "North",
                "state": "Delhi",
                "city": "New Delhi",
                "sales_channel": None,
                "customer_segment": "Consumer",
                "category": "Technology",
                "sub_category": "Phones",
                "product_name": "Phone A",
                "units_sold": 2,
                "unit_price": 1000,
                "unit_cost": 650,
                "discount_pct": None,
                "marketing_spend": None,
                "inventory_days": 10,
                "returned": 0,
                "customer_rating": None,
            },
            {
                "order_id": "ORD-1",
                "order_date": "2024-01-01",
                "ship_date": "2024-01-03",
                "region": "North",
                "state": "Delhi",
                "city": "New Delhi",
                "sales_channel": None,
                "customer_segment": "Consumer",
                "category": "Technology",
                "sub_category": "Phones",
                "product_name": "Phone A",
                "units_sold": 2,
                "unit_price": 1000,
                "unit_cost": 650,
                "discount_pct": None,
                "marketing_spend": None,
                "inventory_days": 10,
                "returned": 0,
                "customer_rating": None,
            },
        ]
    )

    cleaned_df, report = preprocess_data(dataframe)

    assert len(cleaned_df) == 1
    assert report["duplicates_removed"] == 1
    assert cleaned_df["sales_channel"].iloc[0] == "Unknown"
    assert cleaned_df["discount_pct"].isna().sum() == 0
    assert cleaned_df["marketing_spend"].isna().sum() == 0
    assert cleaned_df["customer_rating"].isna().sum() == 0
    assert "unit_price_zscore" in cleaned_df.columns
