import streamlit as st
import pandas as pd
import numpy as np
import ta.momentum as ta_momentum  # pip install ta-lib or use pandas-ta
import matplotlib.pyplot as plt

# Tilson T3 function (from Pine Script)
def tilson_t3(series, length, factor=0.7):
    e1 = series.ewm(span=length, adjust=False).mean()
    e2 = e1.ewm(span=length, adjust=False).mean()
    e3 = e2.ewm(span=length, adjust=False).mean()
    e4 = e3.ewm(span=length, adjust=False).mean()
    e5 = e4.ewm(span=length, adjust=False).mean()
    e6 = e5.ewm(span=length, adjust=False).mean()
    t3 = factor * (e6 - 3*e5 + 3*e4 - e3) + (1 - factor) * (3*e2 - 3*e1 + series)
    return t3

def calculate_ma(df, ma_type, length):
    if ma_type == "SMA":
        return df['Close'].rolling(window=length).mean()
    elif ma_type == "EMA":
        return df['Close'].ewm(span=length, adjust=False).mean()
    elif ma_type == "WMA":
        weights = np.arange(1, length + 1)
        return df['Close'].rolling(length).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)
    elif ma_type == "HullMA":
        wma1 = df['Close'].rolling(window=length//2).mean() * 2
        wma2 = df['Close'].rolling(window=length).mean()
        hull = (wma1 - wma2).rolling(window=int(np.sqrt(length))).mean()
        return hull
    elif ma_type == "VWMA":
        vol = df['Volume'] if 'Volume' in df else pd.Series(1, index=df.index)
        return (df['Close'] * vol).rolling(length).sum() / vol.rolling(length).sum()
    elif ma_type == "RMA":
        return df['Close'].ewm(alpha=1/length, adjust=False).mean()
    elif ma_type == "TEMA":
        ema1 = df['Close'].ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        ema3 = ema2.ewm(span=length, adjust=False).mean()
        return 3*ema1 - 3*ema2 + ema3
    elif ma_type == "Tilson T3":
        factor = 0.7  # default from script
        return tilson_t3(df['Close'], length, factor)
    else:
        return df['Close'].rolling(window=length).mean()  # fallback SMA

def backtest_strategy(df, ma_type, ma_length, rsi_period, rsi_long, rsi_short, sl_pct, tp_pct, qty):
    df['MA'] = calculate_ma(df, ma_type, ma_length)
    df['RSI'] = ta_momentum.RSIIndicator(df['Close'], window=rsi_period).rsi()

    trades = []
    position = None
    entry_price = 0

    for i in range(1, len(df)):
        if position is None:
            # Long entry
            if df['Open'].iloc[i] < df['MA'].iloc[i-1] and df['Close'].iloc[i] > df['MA'].iloc[i] and df['RSI'].iloc[i] > rsi_long:
                position = "LONG"
                entry_price = df['Close'].iloc[i]
                trades.append({
                    "Entry Date": df['Date'].iloc[i],
                    "Entry Price": entry_price,
                    "Position": position,
                    "Qty": qty
                })
            # Short entry
            elif df['Open'].iloc[i] > df['MA'].iloc[i-1] and df['Close'].iloc[i] < df['MA'].iloc[i] and df['RSI'].iloc[i] < rsi_short:
                position = "SHORT"
                entry_price = df['Close'].iloc[i]
                trades.append({
                    "Entry Date": df['Date'].iloc[i],
                    "Entry Price": entry_price,
                    "Position": position,
                    "Qty": qty
                })

        elif position == "LONG":
            # Exit on cross below or SL/TP
            exit_price = df['Close'].iloc[i]
            if df['Open'].iloc[i] > df['MA'].iloc[i-1] and exit_price < df['MA'].iloc[i]:
                reason = "MA Cross"
            elif exit_price <= entry_price * (1 - sl_pct / 100):
                reason = "Stop Loss"
            elif exit_price >= entry_price * (1 + tp_pct / 100):
                reason = "Take Profit"
            else:
                continue
            pnl = (exit_price - entry_price) * qty
            trades[-1].update({
                "Exit Date": df['Date'].iloc[i],
                "Exit Price": exit_price,
                "PNL": pnl,
                "Exit Reason": reason
            })
            position = None

        elif position == "SHORT":
            # Exit on cross above or SL/TP
            exit_price = df['Close'].iloc[i]
            if df['Open'].iloc[i] < df['MA'].iloc[i-1] and exit_price > df['MA'].iloc[i]:
                reason = "MA Cross"
            elif exit_price >= entry_price * (1 + sl_pct / 100):
                reason = "Stop Loss"
            elif exit_price <= entry_price * (1 - tp_pct / 100):
                reason = "Take Profit"
            else:
                continue
            pnl = (entry_price - exit_price) * qty
            trades[-1].update({
                "Exit Date": df['Date'].iloc[i],
                "Exit Price": exit_price,
                "PNL": pnl,
                "Exit Reason": reason
            })
            position = None

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df['Cumulative PNL'] = trades_df['PNL'].cumsum()
    
    return trades_df

st.title("MA Crossover with RSI Confirmation Backtester")

uploaded_file = st.file_uploader("Upload Nifty CSV (Date, Open, High, Low, Close)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])

    ma_type = st.selectbox("MA Type", ["SMA", "EMA", "WMA", "HullMA", "VWMA", "RMA", "TEMA", "Tilson T3"])
    ma_length = st.number_input("MA Length", value=20, min_value=1)
    rsi_period = st.number_input("RSI Period", value=14, min_value=2)
    rsi_long = st.number_input("RSI Long Threshold", value=50, min_value=0, max_value=100)
    rsi_short = st.number_input("RSI Short Threshold", value=50, min_value=0, max_value=100)
    sl_pct = st.number_input("Stop Loss %", value=1.0, min_value=0.0, step=0.1)
    tp_pct = st.number_input("Take Profit %", value=2.0, min_value=0.0, step=0.1)
    qty = st.number_input("Quantity per Trade", value=65, min_value=1)

    if st.button("Run Backtest"):
        trades_df = backtest_strategy(df, ma_type, ma_length, rsi_period, rsi_long, rsi_short, sl_pct, tp_pct, qty)

        st.subheader("Trades Table")
        st.dataframe(trades_df)

        if not trades_df.empty:
            st.subheader("Equity Curve")
            fig, ax = plt.subplots()
            ax.plot(trades_df['Exit Date'], trades_df['Cumulative PNL'], marker='o')
            ax.set_title("Equity Curve")
            ax.set_xlabel("Date")
            ax.set_ylabel("Cumulative PNL")
            st.pyplot(fig)

            # Metrics
            total_pnl = trades_df['PNL'].sum()
            win_rate = (trades_df['PNL'] > 0).mean() * 100 if not trades_df.empty else 0
            max_dd = (trades_df['Cumulative PNL'].cummax() - trades_df['Cumulative PNL']).max() if not trades_df.empty else 0
            st.subheader("Performance Metrics")
            st.write(f"Total PNL: {total_pnl:.2f}")
            st.write(f"Win Rate: {win_rate:.1f}%")
            st.write(f"Max Drawdown: {max_dd:.2f}")
