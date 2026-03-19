import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from yahooquery import Ticker
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="1‑Hr Multi‑Strategy Backtester")

SYMBOLS = {
    "XAUUSD": "GC=F",
    "US100":  "^NDX",
    "US500":  "^GSPC",
}


def fetch_1h_data(ticker_str, days=180):
    now = datetime.now()
    start = now - timedelta(days=days)
    t = Ticker(ticker_str, asynchronous=True)
    df = t.history(period="3mo", interval="1h")
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
    return df


def simulate_mean_reversion_signals(df, sl_scale=1.0, tp_scale=1.5, z_thresh=2.0):
    df = df.copy()
    df["signal"] = 0.0
    df["sl"] = np.nan
    df["tp"] = np.nan
    df["entry_price"] = np.nan

    in_trade = False
    for i in range(1, len(df)):
        close = df.loc[i, "close"]
        low = df.loc[i, "low"]
        high = df.loc[i, "high"]
        z = df.loc[i, "z"]
        rsi = df.loc[i, "rsi"]
        adx = df.loc[i, "adx"]

        long_cond = (
            not in_trade
            and z < -z_thresh
            and rsi < 30
            and adx < 25
        )
        short_cond = (
            not in_trade
            and z > z_thresh
            and rsi > 70
            and adx < 25
        )

        if long_cond or short_cond:
            next_i = i + 1
            if next_i >= len(df):
                continue
            next_close = df.loc[next_i, "close"]
            next_open = df.loc[next_i, "open"]
            if (long_cond and next_close > next_open) or (short_cond and next_close < next_open):
                atr = df.loc[i - 10:i, "high"].max() - df.loc[i - 10:i, "low"].min()
                if atr <= 0:
                    continue
                sl = low if long_cond else high
                tp = close + tp_scale * atr if long_cond else close - tp_scale * atr
                sl = sl - (sl_scale * atr if long_cond else -sl_scale * atr)
                df.loc[next_i, "signal"] = 1.0 if long_cond else -1.0
                df.loc[next_i, "entry_price"] = close
                df.loc[next_i, "sl"] = sl
                df.loc[next_i, "tp"] = tp
                in_trade = True
        elif in_trade:
            prev_entry = df.loc[i - 1, "entry_price"]
            prev_sl = df.loc[i - 1, "sl"]
            prev_tp = df.loc[i - 1, "tp"]
            if pd.isna(prev_entry) or pd.isna(prev_sl) or pd.isna(prev_tp):
                continue
            if df.loc[i - 1, "signal"] > 0:
                if close >= prev_tp:
                    in_trade = False
                elif close <= prev_sl:
                    in_trade = False
            if df.loc[i - 1, "signal"] < 0:
                if close <= prev_tp:
                    in_trade = False
                elif close >= prev_sl:
                    in_trade = False
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

    in_trade = False
    for i in range(20, len(df)):
        close = df.loc[i, "close"]
        low = df.loc[i, "low"]
        high = df.loc[i, "high"]

        if not in_trade and df.loc[i, "trend_up"] and close < df.loc[i, "ma20"]:
            atr = df.loc[i - 10:i, "high"].max() - df.loc[i - 10:i, "low"].min()
            if atr <= 0:
                continue
            sl = low - sl_scale * atr
            tp = close + tp_scale * atr
            df.loc[i, "signal"] = 1.0
            df.loc[i, "entry_price"] = close
            df.loc[i, "sl"] = sl
            df.loc[i, "tp"] = tp
            in_trade = True
        elif not in_trade and df.loc[i, "trend_down"] and close > df.loc[i, "ma20"]:
            atr = df.loc[i - 10:i, "high"].max() - df.loc[i - 10:i, "low"].min()
            if atr <= 0:
                continue
            sl = high + sl_scale * atr
            tp = close - tp_scale * atr
            df.loc[i, "signal"] = -1.0
            df.loc[i, "entry_price"] = close
            df.loc[i, "sl"] = sl
            df.loc[i, "tp"] = tp
            in_trade = True
        elif in_trade:
            prev_entry = df.loc[i - 1, "entry_price"]
            prev_sl = df.loc[i - 1, "sl"]
            prev_tp = df.loc[i - 1, "tp"]
            if pd.isna(prev_entry) or pd.isna(prev_sl) or pd.isna(prev_tp):
                continue
            sign = 1.0 if df.loc[i - 1, "signal"] > 0 else -1.0
            if (sign > 0 and close >= prev_tp) or (sign < 0 and close <= prev_tp):
                in_trade = False
            if (sign > 0 and close <= prev_sl) or (sign < 0 and close >= prev_sl):
                in_trade = False
    return df


def simulate_speed_momentum_signals(df, sl_scale=0.8, tp_scale=2.0):
    df = df.copy()
    df["signal"] = 0.0
    df["sl"] = np.nan
    df["tp"] = np.nan
    df["entry_price"] = np.nan

    vol_ma = df["volume"].rolling(20).mean()
    vol_std = df["volume"].rolling(20).std()
    vol_z = (df["volume"] - vol_ma) / (vol_std + 1e-8)

    in_trade = False
    for i in range(20, len(df)):
        close = df.loc[i, "close"]
        low = df.loc[i, "low"]
        high = df.loc[i, "high"]
        roc = df.loc[i, "roc"]
        v_z = vol_z.iloc[i]

        long_cond = not in_trade and roc > 0.05 and v_z > 1.5
        short_cond = not in_trade and roc < -0.05 and v_z > 1.5

        if long_cond or short_cond:
            atr = df.loc[i - 10:i, "high"].max() - df.loc[i - 10:i, "low"].min()
            if atr <= 0:
                continue
            side = 1.0 if long_cond else -1.0
            sl_val = low - sl_scale * atr if long_cond else high + sl_scale * atr
            tp_val = close + tp_scale * atr if long_cond else close - tp_scale * atr
            df.loc[i, "signal"] = side
            df.loc[i, "entry_price"] = close
            df.loc[i, "sl"] = sl_val
            df.loc[i, "tp"] = tp_val
            in_trade = True
        elif in_trade:
            prev_entry = df.loc[i - 1, "entry_price"]
            prev_sl = df.loc[i - 1, "sl"]
            prev_tp = df.loc[i - 1, "tp"]
            if pd.isna(prev_entry) or pd.isna(prev_sl) or pd.isna(prev_tp):
                continue
            side = 1.0 if df.loc[i - 1, "signal"] > 0 else -1.0
            if (side > 0 and close >= prev_tp) or (side < 0 and close <= prev_tp):
                in_trade = False
            if (side > 0 and close <= prev_sl) or (side < 0 and close >= prev_sl):
                in_trade = False
    return df


def backtest_signals(df, strat_name):
    trades = []
    in_trade = False
    entry_idx = -1

    for i in range(len(df)):
        sig = df.loc[i, "signal"]
        close = df.loc[i, "close"]
        sl = df.loc[i, "sl"]
        tp = df.loc[i, "tp"]
        entry_price = df.loc[i, "entry_price"]

        if sig != 0.0 and not in_trade:
            in_trade = True
            entry_idx = i

        if in_trade:
            if sig > 0:
                if close >= tp and not pd.isna(tp):
                    pnl = (tp - entry_price) / (entry_price - sl)
                    trades.append({
                        "strategy": strat_name,
                        "asset": "N/A",
                        "direction": "long",
                        "status": "win",
                        "pnl_pct": pnl
                    })
                    in_trade = False
                elif close <= sl and not pd.isna(sl):
                    pnl = (sl - entry_price) / (entry_price - sl)
                    trades.append({
                        "strategy": strat_name,
                        "asset": "N/A",
                        "direction": "long",
                        "status": "loss",
                        "pnl_pct": pnl
                    })
                    in_trade = False
            elif sig < 0:
                if close <= tp and not pd.isna(tp):
                    pnl = (entry_price - tp) / (sl - entry_price)
                    trades.append({
                        "strategy": strat_name,
                        "asset": "N/A",
                        "direction": "short",
                        "status": "win",
                        "pnl_pct": pnl
                    })
                    in_trade = False
                elif close >= sl and not pd.isna(sl):
                    pnl = (entry_price - sl) / (sl - entry_price)
                    trades.append({
                        "strategy": strat_name,
                        "asset": "N/A",
                        "direction": "short",
                        "status": "loss",
                        "pnl_pct": pnl
                    })
                    in_trade = False

    if not trades:
        return pd.DataFrame(), 0.0, 0.0, 0.0

    df_trades = pd.DataFrame(trades)
    total_return = df_trades["pnl_pct"].sum()
    wins = df_trades["status"] == "win"
    win_rate = wins.mean() if len(wins) > 0 else 0.0
    avg_rr = df_trades["pnl_pct"].mean() if len(df_trades) > 0 else 0.0

    return df_trades, total_return, avg_rr, win_rate


st.title("📈 1‑Hr Multi‑Strategy Backtester (yahooquery)")

selected_assets = st.multiselect(
    "Select assets",
    list(SYMBOLS.keys()),
    default=["XAUUSD", "US100", "US500"]
)

STRAT_DEFS = {
    "mean_reversion": simulate_mean_reversion_signals,
    "trend": simulate_trend_signals,
    "speed_momentum": simulate_speed_momentum_signals,
}

if st.button("Run Backtest") and selected_assets:
    all_results = []
    all_trades_dfs = []

    for asset_name in selected_assets:
        ticker_str = SYMBOLS[asset_name]
        st.subheader(f"📊 {asset_name} ({ticker_str})")

        data = fetch_1h_data(ticker_str, days=180)
        if data is None or len(data) < 50:
            st.warning(f"Failed to load sufficient data for {asset_name}")
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
            title=f"{asset_name} 1‑Hr",
            xaxis_title="Time",
            yaxis_title="Price",
            width=800,
            height=400
        )
        st.plotly_chart(fig)

        for strat_name, sim_func in STRAT_DEFS.items():
            with st.spinner(f"Backtesting {asset_name} – {strat_name} ..."):
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
                all_trades_dfs.append(df_trades)

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
