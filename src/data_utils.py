import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple

# -------------------------
# GLOBAL CONFIG (UNCHANGED)
# -------------------------
TRADING_START = "09:15"
TRADING_END = "15:30"
INTERVAL = "5min"
STRIKE_STEP = 50
RISK_FREE_RATE = 0.065

np.random.seed(42)

# -------------------------
# DATE UTILS
# -------------------------
def get_last_one_year_dates() -> Tuple[str, str]:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)
    return (
        start_date.strftime("%d-%m-%Y"),
        end_date.strftime("%d-%m-%Y")
    )

# -------------------------
# TRADING TIMESTAMPS
# -------------------------
def generate_trading_timestamps(start_date: str, end_date: str) -> pd.DatetimeIndex:
    start_dt = datetime.strptime(start_date, "%d-%m-%Y")
    end_dt = datetime.strptime(end_date, "%d-%m-%Y")

    trading_days = pd.bdate_range(start_dt, end_dt)

    timestamps = []
    for day in trading_days:
        intraday = pd.date_range(
            f"{day.date()} {TRADING_START}",
            f"{day.date()} {TRADING_END}",
            freq=INTERVAL
        )
        timestamps.extend(intraday)

    return pd.DatetimeIndex(timestamps)

# -------------------------
# RAW SPOT DATA (OHLCV)
# -------------------------
def generate_raw_spot_data(timestamps, start_price=18000):
    n = len(timestamps)

    returns = np.random.normal(0.00002, 0.0018, n)
    price = start_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        "timestamp": timestamps,
        "spot_open": price,
        "spot_high": price * (1 + np.random.uniform(0.0005, 0.002, n)),
        "spot_low":  price * (1 - np.random.uniform(0.0005, 0.002, n)),
        "spot_close": price,
        "spot_volume": np.random.randint(50_000, 500_000, n)
    })

    # inject NaNs (realistic exchange issues)
    for col in ["spot_open", "spot_high", "spot_low", "spot_close", "spot_volume"]:
        nan_idx = np.random.choice(df.index, size=int(0.005 * n), replace=False)
        df.loc[nan_idx, col] = np.nan

    return df

# -------------------------
# RAW FUTURES DATA (OHLCV + OI)
# -------------------------
def generate_raw_futures_data(spot_df):
    df = spot_df.copy()

    basis = np.random.normal(0.002, 0.0007, len(df))

    df["fut_open"] = df["spot_open"] * (1 + basis)
    df["fut_high"] = df["spot_high"] * (1 + basis)
    df["fut_low"]  = df["spot_low"]  * (1 + basis)
    df["fut_close"] = df["spot_close"] * (1 + basis)

    df["fut_volume"] = np.random.randint(200_000, 1_000_000, len(df))
    df["fut_open_interest"] = np.random.randint(2_000_000, 8_000_000, len(df))

    # simulate rollover ambiguity
    df["contract"] = np.where(
        df["timestamp"].dt.day > 20,
        "NIFTY_NEXT_FUT",
        "NIFTY_CURR_FUT"
    )

    # missing OI (common in raw data)
    nan_idx = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    df.loc[nan_idx, "fut_open_interest"] = np.nan

    return df[[
        "timestamp",
        "contract",
        "fut_open",
        "fut_high",
        "fut_low",
        "fut_close",
        "fut_volume",
        "fut_open_interest"
    ]]

# -------------------------
# RAW OPTIONS DATA (ATM ±2, CE & PE)
# -------------------------
def generate_raw_options_data(spot_df):
    rows = []

    for _, row in spot_df.iterrows():
        spot = row["spot_close"]
        ts = row["timestamp"]

        if pd.isna(spot):
            continue

        atm = round(spot / STRIKE_STEP) * STRIKE_STEP
        strikes = [atm - 100, atm - 50, atm, atm + 50, atm + 100]

        base_iv = abs(np.random.normal(0.16, 0.04))

        for strike in strikes:
            for opt_type in ["CE", "PE"]:
                iv = base_iv + (0.02 if opt_type == "PE" else 0)

                # inject bad IV occasionally
                if np.random.rand() < 0.05:
                    iv = np.nan

                rows.append({
                    "timestamp": ts,
                    "opt_expiry": ts + pd.Timedelta(days=7),
                    "opt_strike": strike,
                    "opt_type": opt_type,
                    "opt_ltp": max(1, abs(spot - strike) * 0.5),
                    "opt_iv": iv,
                    "opt_open_interest": np.random.randint(10_000, 300_000),
                    "opt_volume": np.random.randint(0, 15_000)
                })

    return pd.DataFrame(rows)
