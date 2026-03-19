import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from yahooquery import Ticker
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="1-Hr Multi-Strategy Backtester")

SYMBOLS = {
    "XAUUSD": "GC=F",
    "US100": "QQQ",
    "US500": "SPY",
}


def fetch_1h_data(ticker_str, days=180):
    t = Ticker(ticker_str, asynchronous=True)
    df = t.history(period="6mo", interval="1h")
    if df is not None and len(df) > 0:
        df = df.reset_index()
        df["timestamp"] = df["date"].dt.tz_localize(None)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df[["timestamp", "open", "high", "low", "close", "volume"]]
    return None


def add_indicators(df):
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    df["ma20"] = df["close"].rolling(20).mean()
    df["std20"] = df["close"].rolling(20).std()
    df["bb_upper"] = df["ma20"] + 2 * df["std20"]
    df["bb_lower"] = df["ma20"] - 2 * df["std20"]
    df["z"] = (df["close"] - df["ma20"]) / (df["std20"] + 1e-8)
    delta = df["ret"].copy()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-8)
    df["rsi"] = 100 - (100 / (1 + rs))
    up_move = df["high"].diff()
    down_move = df["low"].diff()
    plus_dm = up_move.where(up_move > 0, 0)
    minus_dm = down_move.where(down_move < 0, 0).abs()
    tr = (df["high"] - df["low"]).clip(lower=0)
    df["adx_ma"] = tr.rolling(14).mean()
    df["adx"] = 100 * (abs(df["close"].diff(periods=14)) / (df["adx_ma"] + 1e-8))
    df["roc"] = df["close"].pct_change(12)
    vol_ma = df["volume"].rolling(20).mean()
    vol_std = df["volume"].rolling(20).std()
    df["vol_z"] = (df["volume"] - vol_ma) / (vol_std + 1e-8)
    return df


def simulate_mean_reversion_signals(df, sl_scale=1.2, tp_scale=1.5, z_thresh=2.0):
    df = df.copy()
    df["signal"] = 0.0
    df["sl"] = np.nan
    df["tp"] = np.nan
    df["entry_price"] = np.nan

    for i in range(20, len(df)):
        close = df.loc[i, "close"]
        low = df.loc[i, "low"]
        high = df.loc[i, "high"]
        z = df.loc[i, "z"]
        rsi = df.loc[i, "rsi"]
        adx = df.loc[i, "adx"]

        long_cond = (
            not pd.isna(z)
            and z < -z_thresh
            and rsi < 30
            and adx < 25
        )
        short_cond = (
            not pd.isna(z)
            and z > z_thresh
            and rsi > 70
            and adx < 25
        )

        if long_cond or short_cond:
            atr = df.loc[i - 10:i, "high"].max() - df.loc[i - 10:i, "low"].min()
            if atr <= 1e-6:
                continue

            sl = low - sl_scale * atr if long_cond else high + sl_scale * atr
            tp = close + tp_scale * atr if long_cond else close - tp_scale * atr

            df.loc[i, "signal"] = 1.0 if long_cond else -1.0
            df.loc[i, "entry_price"] = close
            df.loc[i, "sl"] = sl
            df.loc[i, "tp"] = tp

    return df


def simulate_trend_signals(df, sl_scale=1.2, tp_scale=1.2):
    df = df.copy()
    df["signal"] = 0.0
    df["sl"] = np.nan
    df["tp"] = np.nan
    df["entry_price"] = np.nan

    df["ma20"] = df["close"].rolling(20).mean()
    df["trend_up"] = (df["ma20"].diff() > 0).rolling(5).mean() > 0.5
    df["trend_down"] = (df["ma20"].diff() < 0).rolling(5).mean() > 0.5

    for i in range(20, len(df)):
        close = df.loc[i, "close"]
        low = df.loc[i, "low"]
        high = df.loc[i, "high"]
        trend_up = df.loc[i, "trend_up"]
        trend_down = df.loc[i, "trend_down"]

        if (
            df.loc[i, "signal"] == 0.0
            and trend_up
            and close < df.loc[i, "ma20"]
        ):
            atr = df.loc[i - 10:i, "high"].max() - df.loc[i - 10:i, "low"].min()
            if atr <= 1e-6:
                continue
            sl = low - sl_scale * atr
            tp = close + tp_scale * atr
            df.loc[i, "signal"] = 1.0
            df.loc[i, "entry_price"] = close
            df.loc[i, "sl"] = sl
            df.loc[i, "tp"] = tp
        elif (
            df.loc[i, "signal"] == 0.0
            and trend_down
            and close > df.loc[i, "ma20"]
        ):
            atr = df.loc[i - 10:i, "high"].max() - df.loc[i - 10:i, "low"].min()
            if atr <= 1e-6:
                continue
            sl = high + sl_scale * atr
            tp = close - tp_scale * atr
            df.loc[i, "signal"] = -1.0
            df.loc[i, "entry_price"] = close
            df.loc[i, "sl"] = sl
            df.loc[i, "tp"] = tp

    return df


def simulate_speed_momentum_signals(df, sl_scale=0.8, tp_scale=2.0):
    df = df.copy()
    df["signal"] = 0.0
    df["sl"] = np.nan
    df["tp"] = np.nan
    df["entry_price"] = np.nan

    for i in range(20, len(df)):
        close = df.loc[i, "close"]
        low = df.loc[i, "low"]
        high = df.loc[i, "high"]
        roc = df.loc[i, "roc"]
        vol_z = df.loc[i, "vol_z"]

        long_cond = not pd.isna(roc) and roc > 0.01 and vol_z > 1.0
        short_cond = not pd.isna(roc) and roc < -0.01 and vol_z > 1.0

        if long_cond or short_cond:
            atr = df.loc[i - 10:i, "high"].max() - df.loc[i - 10:i, "low"].min()
            if atr <= 1e-6:
                continue

            sl = low - sl_scale * atr if long_cond else high + sl_scale * atr
            tp = close + tp_scale * atr if long_cond else close - tp_scale * atr

            df.loc[i, "signal"] = 1.0 if long_cond else -1.0
            df.loc[i, "entry_price"] = close
            df.loc[i, "sl"] = sl
            df.loc[i, "tp"] = tp

    return df


def backtest_signals(df, strat_name):
    trades = []
    position = None

    for i in range(len(df)):
        sig = df.loc[i, "signal"]
        close = df.loc[i, "close"]
        high = df.loc[i, "high"]
        low = df.loc[i, "low"]
        sl = df.loc[i, "sl"]
        tp = df.loc[i, "tp"]
        entry_price = df.loc[i, "entry_price"]

        if position is None and sig != 0.0:
            position = {
                "direction": "long" if sig > 0 else "short",
                "entry_price": entry_price,
                "sl": sl,
                "tp": tp,
            }
            continue

        if position is None:
            continue

        if position["direction"] == "long":
            hit_sl = not pd.isna(position["sl"]) and low <= position["sl"]
            hit_tp = not pd.isna(position["tp"]) and high >= position["tp"]

            if hit_tp or hit_sl:
                if hit_tp and not hit_sl:
                    pnl = (position["tp"] - position["entry_price"]) / (position["entry_price"] - position["sl"])
                    status = "win"
                    exit_price = position["tp"]
                else:
                    pnl = (position["sl"] - position["entry_price"]) / (position["entry_price"] - position["sl"])
                    status = "loss"
                    exit_price = position["sl"]

                trades.append({
                    "strategy": strat_name,
                    "asset": "N/A",
                    "direction": "long",
                    "status": status,
                    "pnl_pct": pnl
                })
                position = None

        else:
            hit_sl = not pd.isna(position["sl"]) and high >= position["sl"]
            hit_tp = not pd.isna(position["tp"]) and low <= position["tp"]

            if hit_tp or hit_sl:
                if hit_tp and not hit_sl:
                    pnl = (position["entry_price"] - position["tp"]) / (position["sl"] - position["entry_price"])
                    status = "win"
                    exit_price = position["tp"]
                else:
                    pnl = (position["entry_price"] - position["sl"]) / (position["sl"] - position["entry_price"])
                    status = "loss"
                    exit_price = position["sl"]

                trades.append({
                    "strategy": strat_name,
                    "asset": "N/A",
                    "direction": "short",
                    "status": status,
                    "pnl_pct": pnl
                })
                position = None

    if not trades:
        return pd.DataFrame(), 0.0, 0.0, 0.0

    df_trades = pd.DataFrame(trades)
    total_return = df_trades["pnl_pct"].sum()
    wins = df_trades["status"] == "win"
    win_rate = wins.mean() if len(wins) > 0 else 0.0
    avg_rr = df_trades["pnl_pct"].mean() if len(df_trades) > 0 else 0.0

    return df_trades, total_return, avg_rr, win_rate


st.title("📈 1-Hr Multi-Strategy Backtester (yahooquery)")

selected_assets = st.multiselect(
    "Select assets",
    list(SYMBOLS.keys()),
    default=list(SYMBOLS.keys())
)

STRAT_DEFS = {
    "mean_reversion": simulate_mean_reversion_signals,
    "trend": simulate_trend_signals,
    "speed_momentum": simulate_speed_momentum_signals,
}

if st.button("Run Backtest") and selected_assets:
    all_results = []

    for asset_name in selected_assets:
        ticker_str = SYMBOLS[asset_name]
        st.subheader(f"📊 {asset_name} ({ticker_str})")

        data = fetch_1h_data(ticker_str, days=180)
        if data is None or len(data) < 50:
            st.warning(f"Failed to load data for {asset_name}")
            continue

        df = add_indicators(data)

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC"
        ))
        fig.update_layout(
            title=f"{asset_name} 1-Hr",
            xaxis_title="Time",
            yaxis_title="Price",
            width=800,
            height=400
        )
        st.plotly_chart(fig)

        for strat_name, sim_func in STRAT_DEFS.items():
            df_sig = sim_func(df.copy())
            df_trades, total_return, avg_rr, win_rate = backtest_signals(df_sig, strat_name)

            all_results.append({
                "asset": asset_name,
                "strategy": strat_name,
                "total_return": total_return,
                "avg_rr": avg_rr,
                "win_rate": win_rate,
                "n_trades": len(df_trades),
            })

            if len(df_trades) > 0:
                st.write(f"**{strat_name}** → {len(df_trades)} trades (avg RR: {avg_rr:.2f}, WR: {win_rate:.1%})")
            else:
                st.write(f"**{strat_name}** → 0 trades")

    if all_results:
        df_results = pd.DataFrame(all_results)
        st.dataframe(df_results.style.format({
            "total_return": "{:.2%}",
            "avg_rr": "{:.2f}",
            "win_rate": "{:.2%}",
        }))
        pivot = df_results.pivot_table(
            index="strategy",
            values=["total_return", "avg_rr", "win_rate"],
            aggfunc="mean"
        )
        st.markdown("### Overall Strategy Summary")
        st.dataframe(pivot.style.format({
            "total_return": "{:.2%}",
            "avg_rr": "{:.2f}",
            "win_rate": "{:.2%}",
        }))