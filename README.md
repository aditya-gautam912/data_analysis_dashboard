# Retail Analytics Dashboard

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![SQL](https://img.shields.io/badge/SQL-SQLite%20%7C%20PostgreSQL%20%7C%20MySQL-green.svg)](#sql-integration)
[![Tests](https://img.shields.io/badge/Tests-pytest-success.svg)](#run-tests)

Portfolio-ready retail analytics project built with Python, Pandas, NumPy, Matplotlib, Seaborn, SQL, and Streamlit. It demonstrates a complete analytics workflow from raw sales data to business insights, forecast evaluation, and an interactive dashboard.

## Highlights

- end-to-end analytics pipeline using CSV, SQL, pandas, and Streamlit
- real dataset support through external CSV ingestion and schema normalization
- cleaning, preprocessing, feature engineering, and outlier-aware EDA
- forecasting with holdout evaluation using `MAE`, `RMSE`, and `MAPE`
- interactive dashboard with filters, KPI deltas, drilldowns, and CSV export
- tested core modules with `pytest`

## Tech Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- SQLAlchemy with SQLite, PostgreSQL, and MySQL support
- Streamlit
- Pytest

## Quick Start

```bash
pip install -r requirements.txt
python -m src.main
streamlit run src/dashboard.py
```

## What This Project Does

- loads retail sales data from CSV and SQL
- cleans missing values, duplicates, and data types
- engineers business features such as revenue, profit, margin, and shipping delay
- performs exploratory data analysis and trend detection
- evaluates simple forecasting models on a holdout window
- generates exportable charts, reports, and processed datasets

## Business Scenario

This project analyzes multi-region retail sales performance to answer practical business questions:

- Which products, categories, and regions drive the most revenue and profit?
- Where are margins shrinking because of discounts or shipping delays?
- What seasonal patterns appear in daily sales?
- What can the next 30 days of revenue look like using simple forecasting?

## Dataset

The repository includes a realistic sample retail dataset generator. The generated data contains:

- order dates and shipping dates
- product category and sub-category
- region, state, and city
- sales channel and customer segment
- units sold, price, cost, discount
- marketing spend, returns, and ratings

The generator intentionally adds:

- missing values
- duplicate rows
- mixed data quality issues

That makes the project useful for demonstrating real preprocessing work.

The pipeline also supports real retail CSV files through `EXTERNAL_DATA_PATH`.
If you provide a path in `.env`, the loader will prefer that file over the generated sample dataset.

Supported external schemas:

- the project's native retail schema
- Superstore-style CSV files with columns such as `Order ID`, `Order Date`, `Ship Date`, `Segment`, `Category`, `Sub-Category`, `Sales`, `Quantity`, `Discount`, and `Profit`

## Real-World Dataset Sources

You can replace the included sample dataset with a real dataset from:

- Kaggle: Superstore Sales, Retail Sales Forecasting, E-commerce datasets
- Public APIs: World Bank, FRED, open government retail or economic datasets
- Internal databases: MySQL or PostgreSQL transaction tables

Recommended starting point:

- Kaggle "Sample Superstore" for retail analytics
- Kaggle "Global Superstore"
- A sales table from PostgreSQL or MySQL

## Project Structure

```text
data_analysis_dashboard/
|-- data/
|   |-- raw/
|   |-- processed/
|-- outputs/
|   |-- charts/
|   |-- reports/
|-- sql/
|   |-- schema.sql
|   |-- queries.sql
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- data_generation.py
|   |-- data_loader.py
|   |-- preprocessing.py
|   |-- feature_engineering.py
|   |-- eda.py
|   |-- forecasting.py
|   |-- sql_integration.py
|   |-- reporting.py
|   |-- main.py
|   |-- dashboard.py
|-- .env.example
|-- requirements.txt
|-- README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run The Analysis Pipeline

```bash
python -m src.main
```

This will:

- generate the sample dataset if it does not exist
- load data from CSV and SQLite
- clean and preprocess the data
- engineer useful features
- create charts and reports
- export processed data to CSV and Excel
- evaluate forecasting models on a holdout window

## Use A Real Dataset

Set `.env` like this:

```env
DATABASE_URL=sqlite:///data/retail_dashboard.db
EXTERNAL_DATA_PATH=C:/path/to/your/retail_dataset.csv
```

The loader will normalize supported external retail schemas into the project's analytics model.

## Run The Streamlit Dashboard

```bash
streamlit run src/dashboard.py
```

Dashboard additions include:

- interactive filters for region, category, segment, channel, and date range
- daily or monthly trend granularity
- period-over-period KPI comparison
- drilldown tables for region and category performance
- filtered CSV download
- forecast evaluation view

## Run Tests

```bash
python -m pytest
```

## SQL Integration

By default the project uses a local SQLite database for a self-contained demo.

You can also connect to PostgreSQL or MySQL by setting a SQLAlchemy connection string in `.env`:

```env
DATABASE_URL=postgresql+psycopg2://username:password@localhost:5432/retail_analytics
```

or

```env
DATABASE_URL=mysql+pymysql://username:password@localhost:3306/retail_analytics
```

## Outputs

Generated outputs are saved in:

- `data/processed/cleaned_retail_sales.csv`
- `data/processed/featured_retail_sales.csv`
- `outputs/charts/`
- `outputs/reports/`

## Resume Value

This project demonstrates:

- end-to-end analytics workflow design
- professional Python project structure
- SQL plus pandas integration
- practical business KPI analysis
- dashboard development with Streamlit
- forecasting and reporting
