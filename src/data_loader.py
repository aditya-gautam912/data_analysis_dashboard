from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from .config import DATA_SOURCE, LIVE_DATA_PATH, RAW_DATA_PATH, SQLITE_DB_PATH, SQL_DIR
from .data_generation import generate_sample_retail_dataset
from .data_sources.coingecko import fetch_coingecko_data, is_available as coingecko_available
from .dataset_adapter import normalize_retail_dataset


def ensure_sample_dataset(path: Path = RAW_DATA_PATH) -> Path:
    if not path.exists():
        generate_sample_retail_dataset(path)
    return path


def resolve_dataset_path(path: Path | None = None) -> Path:
    external_data_path = os.getenv("EXTERNAL_DATA_PATH")
    if external_data_path:
        external_path = Path(external_data_path).expanduser()
        if external_path.exists():
            return external_path
        raise FileNotFoundError(f"EXTERNAL_DATA_PATH does not exist: {external_path}")

    if path is None:
        path = RAW_DATA_PATH
    ensure_sample_dataset(path)
    return path


def load_live_source() -> pd.DataFrame | None:
    source = (os.getenv("DATA_SOURCE") or DATA_SOURCE).lower().strip()
    if not source:
        return None
    if source == "coingecko":
        if not coingecko_available():
            print("[WARN] CoinGecko API unreachable, falling back to sample data.")
            return None
        print("[LIVE] Fetching data from CoinGecko...")
        df = fetch_coingecko_data(LIVE_DATA_PATH)
        return normalize_retail_dataset(df)
    print(f"[WARN] Unknown DATA_SOURCE '{source}', falling back to sample data.")
    return None


def load_csv_data(path: Path | None = None) -> pd.DataFrame:
    live_df = load_live_source()
    if live_df is not None:
        return live_df
    dataset_path = resolve_dataset_path(path)
    dataframe = pd.read_csv(dataset_path)
    return normalize_retail_dataset(dataframe)


def create_sql_engine() -> tuple:
    database_url = os.getenv("DATABASE_URL", f"sqlite:///{SQLITE_DB_PATH.as_posix()}")
    engine = create_engine(database_url)
    return engine, database_url


def initialize_database(dataframe: pd.DataFrame, table_name: str = "retail_sales") -> str:
    engine, database_url = create_sql_engine()
    schema_text = (SQL_DIR / "schema.sql").read_text(encoding="utf-8")

    with engine.begin() as connection:
        if database_url.startswith("sqlite"):
            connection.exec_driver_sql(schema_text)
            connection.exec_driver_sql(f"DELETE FROM {table_name}")
        dataframe.to_sql(table_name, con=connection, if_exists="append", index=False)

    return database_url


def load_sql_data(query_name: str = "base") -> pd.DataFrame:
    engine, _ = create_sql_engine()
    queries_text = (SQL_DIR / "queries.sql").read_text(encoding="utf-8").strip()
    queries = [query.strip() for query in queries_text.split(";") if query.strip()]

    query_lookup = {
        "base": queries[0],
        "regional_summary": queries[1],
    }
    query = query_lookup[query_name]

    with engine.begin() as connection:
        return pd.read_sql(text(query), connection)
