from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


COINGECKO_BASE = "https://api.coingecko.com/api/v3"
TOP_COINS_COUNT = 10
HISTORICAL_DAYS = 90
REQUEST_DELAY = 1.5


def _get_top_coins() -> list[dict[str, Any]]:
    url = f"{COINGECKO_BASE}/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": TOP_COINS_COUNT,
        "page": 1,
        "sparkline": "false",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _get_coin_history(coin_id: str, days: int = HISTORICAL_DAYS) -> list[list[float]]:
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    prices = data.get("prices", [])
    volumes = data.get("total_volumes", [])
    if not prices:
        return []
    vol_map = {int(v[0] / 1000): v[1] for v in volumes}
    result = []
    for ts_ms, price in prices:
        ts_s = int(ts_ms / 1000)
        volume = vol_map.get(int(ts_ms / 1000 / 1000), 0)
        result.append([ts_s, price, volume])
    return result


def _generate_historical_data(top_coins: list[dict]) -> pd.DataFrame:
    rows = []
    for coin in top_coins:
        coin_id = coin["id"]
        name = coin["name"]
        symbol = coin["symbol"].upper()
        current_price = coin.get("current_price", 0) or 0
        market_cap = coin.get("market_cap", 0) or 0
        total_volume = coin.get("total_volume", 0) or 0
        ath_date_str = coin.get("ath_date", "")

        todays_date = datetime.now().date()
        ath_date = todays_date
        if ath_date_str:
            try:
                ath_date = datetime.fromisoformat(ath_date_str.replace("Z", "+00:00")).date()
            except Exception:
                ath_date = todays_date - timedelta(days=60)

        rows.append(
            {
                "order_id": f"CG-{coin_id}-snapshot",
                "order_date": todays_date,
                "ship_date": todays_date + timedelta(days=1),
                "region": name,
                "state": "Global",
                "city": "Market",
                "category": "Cryptocurrency",
                "sub_category": symbol,
                "product_name": f"{name} ({symbol})",
                "customer_segment": "Retail",
                "sales_channel": "Exchange",
                "units_sold": max(1, int(total_volume / max(current_price, 1))),
                "unit_price": current_price,
                "unit_cost": current_price * 0.85,
                "discount_pct": 0,
                "net_revenue": total_volume,
                "profit": total_volume * 0.15,
                "profit_margin_pct": 15.0,
                "marketing_spend": 0,
                "inventory_days": 7,
                "customer_rating": 4.5,
                "returned": 0,
                "ship_delay_days": 0,
                "year_month": todays_date.strftime("%Y-%m"),
                "day_of_week": todays_date.weekday(),
                "source": "coingecko_live",
            }
        )

        try:
            history = _get_coin_history(coin_id)
            for ts_s, price, volume in history:
                date = datetime.utcfromtimestamp(ts_s).date()
                daily_volume = max(volume, 0)
                daily_price = max(price, 0.01)
                daily_units = max(1, int(daily_volume / daily_price))
                profit_margin = np.random.uniform(5, 25)

                rows.append(
                    {
                        "order_id": f"CG-{coin_id}-{ts_s}",
                        "order_date": date,
                        "ship_date": date + timedelta(days=1),
                        "region": name,
                        "state": "Global",
                        "city": "Market",
                        "category": "Cryptocurrency",
                        "sub_category": symbol,
                        "product_name": f"{name} ({symbol})",
                        "customer_segment": "Retail",
                        "sales_channel": "Exchange",
                        "units_sold": daily_units,
                        "unit_price": round(daily_price, 2),
                        "unit_cost": round(daily_price * 0.85, 2),
                        "discount_pct": 0,
                        "net_revenue": round(daily_volume, 2),
                        "profit": round(daily_volume * profit_margin / 100, 2),
                        "profit_margin_pct": round(profit_margin, 2),
                        "marketing_spend": 0,
                        "inventory_days": 7,
                        "customer_rating": round(np.random.uniform(3.5, 5.0), 1),
                        "returned": int(np.random.poisson(0.5)),
                        "ship_delay_days": 0,
                        "year_month": date.strftime("%Y-%m"),
                        "day_of_week": date.weekday(),
                        "source": "coingecko_live",
                    }
                )
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            print(f"  Skipping history for {name}: {e}")

    return pd.DataFrame(rows)


def fetch_coingecko_data(output_path: str | Path | None = None) -> pd.DataFrame:
    print("Fetching top coins from CoinGecko...")
    top_coins = _get_top_coins()
    print(f"  Got {len(top_coins)} coins. Fetching 90-day history for each...")

    df = _generate_historical_data(top_coins)
    print(f"  Generated {len(df)} rows of live crypto data.")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"  Saved to {output_path}")

    return df


def is_available() -> bool:
    try:
        resp = requests.get(f"{COINGECKO_BASE}/ping", timeout=10)
        return resp.status_code == 200
    except Exception:
        return False
