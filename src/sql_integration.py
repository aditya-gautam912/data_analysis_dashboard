from __future__ import annotations

import pandas as pd

from .data_loader import load_sql_data


def extract_sales_from_sql() -> pd.DataFrame:
    return load_sql_data(query_name="base")


def regional_channel_summary() -> pd.DataFrame:
    return load_sql_data(query_name="regional_summary")
