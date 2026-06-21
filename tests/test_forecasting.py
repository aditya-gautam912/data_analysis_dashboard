import numpy as np
import pandas as pd

from src.forecasting import evaluate_forecasts, prepare_daily_revenue, train_test_split_time_series


def test_prepare_daily_revenue_builds_moving_average():
    dataframe = pd.DataFrame(
        {
            "order_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
            "net_revenue": [100.0, 150.0, 200.0],
        }
    )

    daily_df = prepare_daily_revenue(dataframe)

    assert len(daily_df) == 2
    assert daily_df["net_revenue"].tolist() == [250.0, 200.0]
    assert "moving_average_14" in daily_df.columns


def test_evaluate_forecasts_returns_metrics_for_two_models():
    dataframe = pd.DataFrame(
        {
            "order_date": pd.date_range("2024-01-01", periods=60, freq="D"),
            "net_revenue": np.linspace(100, 500, 60),
        }
    )

    train_df, test_df = train_test_split_time_series(dataframe, test_days=15)
    evaluation_df, metrics_df = evaluate_forecasts(dataframe, test_days=15)

    assert len(train_df) == 45
    assert len(test_df) == 15
    assert len(evaluation_df) == 15
    assert set(metrics_df["model"]) == {"Moving Average", "Linear Regression", "ARIMA", "Ensemble"}
    assert {"mae", "rmse", "mape_pct"}.issubset(metrics_df.columns)
