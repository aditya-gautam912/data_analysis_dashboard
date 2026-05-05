from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
REPORTS_DIR = OUTPUTS_DIR / "reports"
SQL_DIR = BASE_DIR / "sql"

RAW_DATA_PATH = RAW_DIR / "retail_sales_sample.csv"
CLEAN_DATA_PATH = PROCESSED_DIR / "cleaned_retail_sales.csv"
FEATURED_DATA_PATH = PROCESSED_DIR / "featured_retail_sales.csv"
SQLITE_DB_PATH = DATA_DIR / "retail_dashboard.db"
FORECAST_METRICS_PATH = REPORTS_DIR / "forecast_metrics.csv"


def load_environment() -> None:
    load_dotenv(BASE_DIR / ".env")


def ensure_directories() -> None:
    for directory in [RAW_DIR, PROCESSED_DIR, CHARTS_DIR, REPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
