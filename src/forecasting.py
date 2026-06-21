from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA


def prepare_daily_revenue(dataframe: pd.DataFrame) -> pd.DataFrame:
    daily = dataframe.groupby("order_date", as_index=False)["net_revenue"].sum().sort_values("order_date")
    daily["moving_average_14"] = daily["net_revenue"].rolling(window=14, min_periods=1).mean()
    return daily


def simple_linear_forecast(daily_revenue: pd.DataFrame, future_days: int = 30) -> pd.DataFrame:
    return _ensemble_forecast(daily_revenue, future_days)


def train_test_split_time_series(daily_revenue: pd.DataFrame, test_days: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(daily_revenue) <= test_days:
        raise ValueError("Not enough observations to create a train/test split.")
    train_df = daily_revenue.iloc[:-test_days].copy()
    test_df = daily_revenue.iloc[-test_days:].copy()
    return train_df, test_df


def _moving_average_predictions(train_df: pd.DataFrame, test_df: pd.DataFrame, window: int = 14) -> np.ndarray:
    history = train_df["net_revenue"].tolist()
    predictions = []
    for actual in test_df["net_revenue"]:
        recent_values = history[-window:] if len(history) >= window else history
        prediction = float(np.mean(recent_values))
        predictions.append(max(prediction, 0.0))
        history.append(float(actual))
    return np.array(predictions)


def _linear_regression_predictions(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    train_index = np.arange(len(train_df))
    test_index = np.arange(len(train_df), len(train_df) + len(test_df))
    slope, intercept = np.polyfit(train_index, train_df["net_revenue"], 1)
    predictions = intercept + slope * test_index
    return np.maximum(predictions, 0.0)


def _arima_predictions(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    try:
        model = ARIMA(train_df["net_revenue"], order=(1, 1, 1))
        fitted = model.fit()
        predictions = fitted.forecast(steps=len(test_df))
        return np.maximum(predictions.values, 0.0)
    except Exception:
        return _linear_regression_predictions(train_df, test_df)


def _arima_forecast(daily_revenue: pd.DataFrame, future_days: int = 30) -> pd.DataFrame:
    forecast_df = daily_revenue.copy().reset_index(drop=True)
    last_date = forecast_df["order_date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=future_days, freq="D")
    try:
        model = ARIMA(forecast_df["net_revenue"], order=(1, 1, 1))
        fitted = model.fit()
        future_values = fitted.forecast(steps=future_days)
        future_values = np.maximum(future_values.values, 0.0)
    except Exception:
        slope, intercept = np.polyfit(np.arange(len(forecast_df)), forecast_df["net_revenue"], 1)
        future_index = np.arange(len(forecast_df), len(forecast_df) + future_days)
        future_values = np.maximum(intercept + slope * future_index, 0.0)

    return pd.DataFrame({"order_date": future_dates, "forecast_revenue": future_values})


def _ensemble_forecast(daily_revenue: pd.DataFrame, future_days: int = 30) -> pd.DataFrame:
    forecast_df = daily_revenue.copy().reset_index(drop=True)
    last_date = forecast_df["order_date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=future_days, freq="D")

    train_df, test_df = train_test_split_time_series(daily_revenue, test_days=min(30, len(daily_revenue) // 5))
    test_preds = {
        "ma": _moving_average_predictions(train_df, test_df),
        "lr": _linear_regression_predictions(train_df, test_df),
        "arima": _arima_predictions(train_df, test_df),
    }
    actual = test_df["net_revenue"].values
    rmse_scores = {}
    for name, preds in test_preds.items():
        rmse_scores[name] = float(np.sqrt(np.mean((actual - preds) ** 2)))

    total_inv_rmse = sum(1.0 / max(s, 1e-10) for s in rmse_scores.values())
    weights = {name: (1.0 / max(rmse_scores[name], 1e-10)) / total_inv_rmse for name in rmse_scores}

    last_revenue = forecast_df["net_revenue"].iloc[-1]
    recent_avg = forecast_df["net_revenue"].tail(14).mean()

    n = len(forecast_df)
    lr_slope, lr_intercept = np.polyfit(np.arange(n), forecast_df["net_revenue"], 1)
    lr_future = lr_intercept + lr_slope * np.arange(n, n + future_days)
    lr_future = np.maximum(lr_future, 0.0)

    try:
        arima_model = ARIMA(forecast_df["net_revenue"], order=(1, 1, 1))
        arima_fitted = arima_model.fit()
        arima_future = np.maximum(arima_fitted.forecast(steps=future_days).values, 0.0)
    except Exception:
        arima_future = lr_future

    history = forecast_df["net_revenue"].tolist()
    ma_future = np.zeros(future_days)
    for i in range(future_days):
        window = history[-14:] if len(history) >= 14 else history
        pred = float(np.mean(window))
        ma_future[i] = max(pred, 0.0)
        history.append(ma_future[i])

    ensemble = weights["ma"] * ma_future + weights["lr"] * lr_future + weights["arima"] * arima_future

    convergence = np.linspace(1.0, 0.0, min(14, future_days))
    ensemble[: len(convergence)] = (
        convergence * (last_revenue if np.isnan(ensemble[0]) else ensemble[: len(convergence)])
        + (1 - convergence) * recent_avg
    )

    return pd.DataFrame({"order_date": future_dates, "forecast_revenue": np.maximum(ensemble, 0.0)})


def evaluate_forecasts(daily_revenue: pd.DataFrame, test_days: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split_time_series(daily_revenue, test_days=test_days)

    evaluation_df = test_df[["order_date", "net_revenue"]].copy()
    evaluation_df["moving_average_prediction"] = _moving_average_predictions(train_df, test_df)
    evaluation_df["linear_regression_prediction"] = _linear_regression_predictions(train_df, test_df)
    evaluation_df["arima_prediction"] = _arima_predictions(train_df, test_df)

    eval_pred_cols = ["moving_average_prediction", "linear_regression_prediction", "arima_prediction"]
    evaluation_df["ensemble_prediction"] = evaluation_df[eval_pred_cols].mean(axis=1)

    metrics = []
    for model_column, model_name in [
        ("moving_average_prediction", "Moving Average"),
        ("linear_regression_prediction", "Linear Regression"),
        ("arima_prediction", "ARIMA"),
        ("ensemble_prediction", "Ensemble"),
    ]:
        error = evaluation_df["net_revenue"] - evaluation_df[model_column]
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(error**2)))
        mape = float(
            np.mean(
                np.abs(error) / np.clip(np.abs(evaluation_df["net_revenue"]), a_min=1.0, a_max=None)
            )
            * 100
        )
        metrics.append({"model": model_name, "mae": round(mae, 2), "rmse": round(rmse, 2), "mape_pct": round(mape, 2)})

    metrics_df = pd.DataFrame(metrics).sort_values("rmse").reset_index(drop=True)
    return evaluation_df, metrics_df


def plot_forecast(daily_revenue: pd.DataFrame, future_forecast: pd.DataFrame, output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 6))
    sns.lineplot(data=daily_revenue, x="order_date", y="net_revenue", label="Actual Revenue", linewidth=1.4)
    sns.lineplot(data=daily_revenue, x="order_date", y="moving_average_14", label="14-Day Moving Average", linewidth=2)
    sns.lineplot(data=future_forecast, x="order_date", y="forecast_revenue", label="Forecast", linestyle="--", linewidth=2)
    plt.title("Revenue Forecast")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.tight_layout()

    forecast_path = output_dir / "forecast_revenue.png"
    plt.savefig(forecast_path, dpi=200)
    plt.close()
    return str(forecast_path)


def plot_forecast_backtest(evaluation_df: pd.DataFrame, output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 6))
    sns.lineplot(data=evaluation_df, x="order_date", y="net_revenue", label="Actual", linewidth=2)
    sns.lineplot(
        data=evaluation_df,
        x="order_date",
        y="moving_average_prediction",
        label="Moving Average Prediction",
        linestyle="--",
        linewidth=1.8,
    )
    sns.lineplot(
        data=evaluation_df,
        x="order_date",
        y="linear_regression_prediction",
        label="Linear Regression Prediction",
        linestyle=":",
        linewidth=2,
    )
    if "arima_prediction" in evaluation_df.columns:
        sns.lineplot(
            data=evaluation_df,
            x="order_date",
            y="arima_prediction",
            label="ARIMA Prediction",
            linestyle="-.",
            linewidth=2,
        )
    if "ensemble_prediction" in evaluation_df.columns:
        sns.lineplot(
            data=evaluation_df,
            x="order_date",
            y="ensemble_prediction",
            label="Ensemble Prediction",
            linestyle=(0, (3, 1, 1, 1)),
            linewidth=2.5,
        )
    plt.title("Forecast Backtest on Holdout Window")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.tight_layout()

    backtest_path = output_dir / "forecast_backtest.png"
    plt.savefig(backtest_path, dpi=200)
    plt.close()
    return str(backtest_path)
