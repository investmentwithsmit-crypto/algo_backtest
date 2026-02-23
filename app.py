import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import re

def calculate_rsi(df, period=14):
    """Calculate RSI if not present in CSV"""
    delta = df['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_supertrend(df, period=10, multiplier=3.0):
    """Calculate Supertrend"""
    hl2 = (df['High'] + df['Low']) / 2
    tr = pd.concat([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift(1)),
        abs(df['Low'] - df['Close'].shift(1))
    ], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()

    upper_basic = hl2 + (multiplier * atr)
    lower_basic = hl2 - (multiplier * atr)

    upper = upper_basic.copy()
    lower = lower_basic.copy()
    supertrend = pd.Series(np.nan, index=df.index)
    direction = pd.Series(0, index=df.index)

    for i in range(1, len(df)):
        if df['Close'].iloc[i-1] > upper.iloc[i-1]:
            upper.iloc[i] = min(upper_basic.iloc[i], upper.iloc[i-1])
        else:
            upper.iloc[i] = upper_basic.iloc[i]

        if df['Close'].iloc[i-1] < lower.iloc[i-1]:
            lower.iloc[i] = max(lower_basic.iloc[i], lower.iloc[i-1])
        else:
            lower.iloc[i] = lower_basic.iloc[i]

        if df['Close'].iloc[i] > upper.iloc[i]:
            direction.iloc[i] = 1
            supertrend.iloc[i] = lower.iloc[i]
        else:
            direction.iloc[i] = -1
            supertrend.iloc[i] = upper.iloc[i]

        if i > 1 and direction.iloc[i] == direction.iloc[i-1]:
            if direction.iloc[i] == 1:
                supertrend.iloc[i] = max(supertrend.iloc[i], supertrend.iloc[i-1])
            else:
                supertrend.iloc[i] = min(supertrend.iloc[i], supertrend.iloc[i-1])

    return supertrend, direction

def backtest_strategy(df, low_rsi=20.0, high_rsi=80.0, qty=65, pt_pct=2.0, sl_pct=1.0, use_st_filter=False, allow_long=True, allow_short=True, st_period=10, st_mult=3.0):
    """Run backtest on the strategy"""
    trades = []
    equity = [0.0]
    position = None
    entry_idx = entry_price = entry_date = None
    trade_high_water = trade_max_dd = 0.0

    for i in range(1, len(df)):
        prev_rsi = df['RSI'].iloc[i-1]
        curr_rsi = df['RSI'].iloc[i]
        curr_close = df['Close'].iloc[i]
        curr_date = df['Date'].iloc[i]
        st_dir = df['Supertrend_Dir'].iloc[i] if 'Supertrend_Dir' in df else 0

        unreal_pnl = 0.0
        if position == "LONG":
            unreal_pnl = (curr_close - entry_price) * qty
        elif position == "SHORT":
            unreal_pnl = (entry_price - curr_close) * qty

        current_equity = equity[-1] + unreal_pnl
        equity.append(current_equity)

        if position:
            if unreal_pnl > trade_high_water:
                trade_high_water = unreal_pnl
                trade_max_dd = 0.0
            current_dd = trade_high_water - unreal_pnl
            if current_dd > trade_max_dd:
                trade_max_dd = current_dd

        # Stop Loss
        if sl_pct > 0 and position:
            if (position == "LONG" and curr_close <= entry_price * (1 - sl_pct / 100)) or \
               (position == "SHORT" and curr_close >= entry_price * (1 + sl_pct / 100)):
                exit_price = curr_close
                pnl = (exit_price - entry_price) * qty if position == "LONG" else (entry_price - exit_price) * qty
                points = exit_price - entry_price if position == "LONG" else entry_price - exit_price
                bars = i - entry_idx
                trades[-1].update({
                    "Exit Date": curr_date, "Exit Price": exit_price, "PNL": pnl,
                    "Points Captured": points, "Bars Held": bars,
                    "Exit Reason": "Stop Loss", "Max Drawdown": trade_max_dd
                })
                equity[-1] = equity[-2] + pnl
                position = None
                trade_high_water = trade_max_dd = 0.0
                continue

        # Profit Target
        if pt_pct > 0 and position:
            if (position == "LONG" and curr_close >= entry_price * (1 + pt_pct / 100)) or \
               (position == "SHORT" and curr_close <= entry_price * (1 - pt_pct / 100)):
                exit_price = curr_close
                pnl = (exit_price - entry_price) * qty if position == "LONG" else (entry_price - exit_price) * qty
                points = exit_price - entry_price if position == "LONG" else entry_price - exit_price
                bars = i - entry_idx
                trades[-1].update({
                    "Exit Date": curr_date, "Exit Price": exit_price, "PNL": pnl,
                    "Points Captured": points, "Bars Held": bars,
                    "Exit Reason": "Profit Target", "Max Drawdown": trade_max_dd
                })
                equity[-1] = equity[-2] + pnl
                position = None
                trade_high_water = trade_max_dd = 0.0
                continue

        # RSI Reversal exit
        if position == "LONG" and prev_rsi > high_rsi and curr_rsi <= high_rsi:
            exit_price = curr_close
            pnl = (exit_price - entry_price) * qty
            points = exit_price - entry_price
            bars = i - entry_idx
            trades[-1].update({
                "Exit Date": curr_date, "Exit Price": exit_price, "PNL": pnl,
                "Points Captured": points, "Bars Held": bars,
                "Exit Reason": "RSI Reversal", "Max Drawdown": trade_max_dd
            })
            equity[-1] = equity[-2] + pnl
            position = None
            trade_high_water = trade_max_dd = 0.0

        elif position == "SHORT" and prev_rsi < low_rsi and curr_rsi >= low_rsi:
            exit_price = curr_close
            pnl = (entry_price - exit_price) * qty
            points = entry_price - exit_price
            bars = i - entry_idx
            trades[-1].update({
                "Exit Date": curr_date, "Exit Price": exit_price, "PNL": pnl,
                "Points Captured": points, "Bars Held": bars,
                "Exit Reason": "RSI Reversal", "Max Drawdown": trade_max_dd
            })
            equity[-1] = equity[-2] + pnl
            position = None
            trade_high_water = trade_max_dd = 0.0

        # Entry
        else:
            long_condition = prev_rsi < low_rsi and curr_rsi >= low_rsi
            short_condition = prev_rsi > high_rsi and curr_rsi <= high_rsi

            if 'Supertrend_Dir' in df:
                st_dir = df['Supertrend_Dir'].iloc[i]
                if use_st_filter:
                    long_condition = long_condition and (st_dir == 1)
                    short_condition = short_condition and (st_dir == -1)
            else:
                st_dir = 0

            if allow_long and long_condition:
                position = "LONG"
                entry_idx = i
                entry_price = curr_close
                entry_date = curr_date
                trades.append({
                    "Entry Date": entry_date,
                    "Entry Price": entry_price,
                    "Position": "LONG",
                    "Qty": qty,
                    "Supertrend": "Bullish" if st_dir == 1 else "Bearish" if st_dir == -1 else "—",
                    "Max Drawdown": 0.0
                })
                trade_high_water = trade_max_dd = 0.0

            elif allow_short and short_condition:
                position = "SHORT"
                entry_idx = i
                entry_price = curr_close
                entry_date = curr_date
                trades.append({
                    "Entry Date": entry_date,
                    "Entry Price": entry_price,
                    "Position": "SHORT",
                    "Qty": qty,
                    "Supertrend": "Bullish" if st_dir == 1 else "Bearish" if st_dir == -1 else "—",
                    "Max Drawdown": 0.0
                })
                trade_high_water = trade_max_dd = 0.0

    # Close open position if any
    if position and trades:
        exit_price = df['Close'].iloc[-1]
        exit_date = df['Date'].iloc[-1]
        bars = len(df) - 1 - entry_idx
        if position == "LONG":
            pnl = (exit_price - entry_price) * qty
            points = exit_price - entry_price
            reason = "End of Data"
        else:
            pnl = (entry_price - exit_price) * qty
            points = entry_price - exit_price
            reason = "End of Data"

        trades[-1].update({
            "Exit Date": exit_date,
            "Exit Price": exit_price,
            "PNL": pnl,
            "Points Captured": points,
            "Bars Held": bars,
            "Exit Reason": reason,
            "Max Drawdown": trade_max_dd
        })
        equity[-1] = equity[-2] + pnl

    return trades, equity

def calculate_summary(trades, equity):
    closed = [t for t in trades if "PNL" in t and t["PNL"] is not None]
    if not closed:
        return {
            "Total Trades": 0,
            "Win Rate": 0.0,
            "Total PNL": 0.0,
            "Avg PNL/Trade": 0.0,
            "Max Win": 0.0,
            "Max Loss": 0.0,
            "Worst Per-Trade Max DD": 0.0,
            "Max Drawdown": 0.0,
            "Max DD %": 0.0
        }

    pnls = [t["PNL"] for t in closed]
    total_pnl = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    win_rate = wins / len(pnls) * 100 if pnls else 0
    avg_pnl = total_pnl / len(pnls) if pnls else 0
    max_win = max((p for p in pnls if p > 0), default=0)
    max_loss = min((p for p in pnls if p < 0), default=0)

    peak = equity[0]
    max_dd = 0
    for eq in equity:
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd
    max_dd_pct = (max_dd / peak * 100) if peak != 0 else 0

    worst_dd = max((t.get("Max Drawdown", 0) for t in closed), default=0)

    return {
        "Total Trades": len(closed),
        "Win Rate": win_rate,
        "Total PNL": total_pnl,
        "Avg PNL/Trade": avg_pnl,
        "Max Win": max_win,
        "Max Loss": max_loss,
        "Worst Per-Trade Max DD": worst_dd,
        "Max Drawdown": max_dd,
        "Max DD %": max_dd_pct
    }

# Streamlit App
st.title("RSI + Supertrend Backtester")

uploaded_file = st.file_uploader("Upload Hourly Candle CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("CSV Preview:")
    st.dataframe(df.head())

    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        low_rsi = st.number_input("Oversold RSI", value=20.0)
        high_rsi = st.number_input("Overbought RSI", value=80.0)
        qty = st.number_input("Quantity", value=65)
        pt_pct = st.number_input("Profit Target %", value=2.0)
        sl_pct = st.number_input("Stop Loss %", value=1.0)
    with col2:
        use_st_filter = st.checkbox("Use Supertrend Filter")
        allow_long = st.checkbox("Allow Long Trades", value=True)
        allow_short = st.checkbox("Allow Short Trades", value=True)
        st_period = st.number_input("Supertrend Period", value=10)
        st_mult = st.number_input("Supertrend Multiplier", value=3.0)

    if st.button("Run Backtest"):
        # Process Date
        def parse_date(s):
            s_clean = re.split(r'\s*\(', s)[0].strip()
            dt = pd.to_datetime(s_clean, utc=True, errors='coerce')
            if pd.isna(dt):
                st.error(f"Invalid date: {s}")
                return None
            return dt.tz_convert('Asia/Kolkata') if dt.tzinfo else dt

        df['Date'] = df['Date'].apply(parse_date)
        df = df.sort_values('Date').reset_index(drop=True)

        if 'RSI' not in df.columns:
            st.info("Calculating RSI...")
            df['RSI'] = calculate_rsi(df)

        if use_st_filter:
            st.info("Calculating Supertrend...")
            _, direction = calculate_supertrend(df, st_period, st_mult)
            df['Supertrend_Dir'] = direction

        trades, equity = backtest_strategy(
            df, low_rsi, high_rsi, qty, pt_pct, sl_pct,
            use_st_filter, allow_long, allow_short, st_period, st_mult
        )

        # Display Trades
        st.subheader("Trades")
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df = trades_df[[ "Entry Date", "Entry Price", "Position", "Qty", "Exit Date", "Exit Price",
                                    "Points Captured", "PNL", "Bars Held", "Max Drawdown", "Supertrend", "Exit Reason" ]]
            st.dataframe(trades_df, use_container_width=True)
        else:
            st.info("No trades generated.")

        # Display Summary
        st.subheader("Performance Summary")
        summary = calculate_summary(trades, equity)
        summary_df = pd.DataFrame(list(summary.items()), columns=["Metric", "Value"])
        st.table(summary_df)
else:
    st.info("Upload a CSV to start. Required columns: Date, Open, High, Low, Close (RSI optional).")
