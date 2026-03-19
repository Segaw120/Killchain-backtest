import streamlit as st
import pandas as pd
import numpy as np
from itertools import product
from yahooquery import Ticker

st.set_page_config(layout="wide", page_title="NQ Trend Grid Optimizer")

SYMBOL = "NQ=F"
INTERVAL = "1h"
PERIOD = "6mo"


@st.cache_data
def fetch_data():
    t = Ticker(SYMBOL)
    df = t.history(period=PERIOD, interval=INTERVAL)
    df = df.reset_index()
    df["timestamp"] = df["date"].dt.tz_localize(None)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def add_indicators(df, ma_fast, ma_slow):
    df = df.copy()
    df["ma_fast"] = df["close"].rolling(ma_fast).mean()
    df["ma_slow"] = df["close"].rolling(ma_slow).mean()
    df["ret"] = df["close"].pct_change()

    delta = df["ret"]
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-8)
    df["rsi"] = 100 - (100 / (1 + rs))

    df["atr"] = df["high"].rolling(10).max() - df["low"].rolling(10).min()
    return df


def simulate_trend(df, rsi_low, rsi_high, sl_scale, rr):
    position = None
    trades = []

    for i in range(50, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        if pd.isna(row["ma_fast"]) or pd.isna(row["ma_slow"]) or pd.isna(row["atr"]):
            continue

        uptrend = row["ma_fast"] > row["ma_slow"] and row["ma_fast"] > prev["ma_fast"]
        downtrend = row["ma_fast"] < row["ma_slow"] and row["ma_fast"] < prev["ma_fast"]

        if position is None:
            if uptrend and prev["close"] <= prev["ma_fast"] and rsi_low <= row["rsi"] <= rsi_high:
                if row["close"] > row["ma_fast"]:
                    atr = row["atr"]
                    sl = df.iloc[i-5:i]["low"].min() - sl_scale * atr
                    tp = row["close"] + rr * (row["close"] - sl)
                    position = {
                        "dir": "long",
                        "entry": row["close"],
                        "sl": sl,
                        "tp": tp
                    }

            elif downtrend and prev["close"] >= prev["ma_fast"] and rsi_low <= row["rsi"] <= rsi_high:
                if row["close"] < row["ma_fast"]:
                    atr = row["atr"]
                    sl = df.iloc[i-5:i]["high"].max() + sl_scale * atr
                    tp = row["close"] - rr * (sl - row["close"])
                    position = {
                        "dir": "short",
                        "entry": row["close"],
                        "sl": sl,
                        "tp": tp
                    }

        else:
            high = row["high"]
            low = row["low"]

            if position["dir"] == "long":
                if low <= position["sl"]:
                    trades.append(0)
                    position = None
                elif high >= position["tp"]:
                    trades.append(1)
                    position = None

            else:
                if high >= position["sl"]:
                    trades.append(0)
                    position = None
                elif low <= position["tp"]:
                    trades.append(1)
                    position = None

    if len(trades) == 0:
        return 0.0, 0.0

    win_rate = float(np.mean(trades))
    trades_per_week = len(trades) / (len(df) / 24 / 7)

    return win_rate, trades_per_week


def run_grid_search(df, progress_bar):
    ma_fast_vals = [10, 20, 30]
    ma_slow_vals = [40, 50, 60]
    rsi_ranges = [(35, 55), (40, 60), (45, 65)]
    sl_scales = [0.8, 1.0, 1.2]
    rr_vals = [1.5, 2.0, 2.5]

    combos = list(product(ma_fast_vals, ma_slow_vals, rsi_ranges, sl_scales, rr_vals))
    results = []

    best = None
    best_score = -1.0

    for idx, (ma_fast, ma_slow, (rsi_low, rsi_high), sl_scale, rr) in enumerate(combos):
        if ma_fast >= ma_slow:
            continue

        df_ind = add_indicators(df, ma_fast, ma_slow)
        win_rate, trades_per_week = simulate_trend(df_ind, rsi_low, rsi_high, sl_scale, rr)

        result = {
            "ma_fast": ma_fast,
            "ma_slow": ma_slow,
            "rsi_low": rsi_low,
            "rsi_high": rsi_high,
            "sl_scale": sl_scale,
            "rr": rr,
            "win_rate": win_rate,
            "trades_per_week": trades_per_week
        }

        results.append(result)

        if trades_per_week >= 1 and win_rate > best_score:
            best_score = win_rate
            best = result

        progress_bar.progress((idx + 1) / len(combos))

    return pd.DataFrame(results), best


st.title("📊 Nasdaq Futures (NQ) Trend Strategy Optimizer")

df = fetch_data()
st.write(f"Loaded {len(df)} rows")

if st.button("Run Optimization"):
    progress_bar = st.progress(0)
    results_df, best = run_grid_search(df, progress_bar)

    st.subheader("All Results")
    st.dataframe(results_df.sort_values("win_rate", ascending=False))

    if best:
        st.subheader("🏆 Best Parameter Set (Qualified)")
        st.json(best)
    else:
        st.warning("No parameter set met the ≥1 trade/week requirement")
