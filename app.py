import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from typing import Callable

# ────────────────────────────────────────────────
# Date Parser
# ────────────────────────────────────────────────
def parse_date(date_str: str) → datetime.datetime:
    # Example: "Mon Feb 09 2026 09:15:00 GMT+0530 (India Standard Time)"
    try:
        if '(' in date_str:
            date_str = date_str.split(' (')[0]
        dt = datetime.datetime.strptime(date_str, "%a %b %d %Y %H:%M:%S %Z%z")
        return dt
    except ValueError:
        st.error(f"Could not parse date: {date_str}")
        return None

# ────────────────────────────────────────────────
# Moving Average Functions
# ────────────────────────────────────────────────
def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length).mean()

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def wma(series: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def hullma(series: pd.Series, length: int) -> pd.Series:
    wma_half = wma(series, length // 2)
    wma_full = wma(series, length)
    hull = wma(2 * wma_half - wma_full, int(np.sqrt(length)))
    return hull

def vwma(series: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    return (series * volume).rolling(window=length).sum() / volume.rolling(window=length).sum()

def rma(series: pd.Series, length: int) -> pd.Series:
    alpha = 1 / length
    rma_series = pd.Series(np.nan, index=series.index)
    if len(series) >= length:
        rma_series.iloc[length-1] = series.iloc[:length].mean()
        for i in range(length, len(series)):
            rma_series.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * rma_series.iloc[i-1]
    return rma_series

def tema(series: pd.Series, length: int) -> pd.Series:
    ema1 = ema(series, length)
    ema2 = ema(ema1, length)
    ema3 = ema(ema2, length)
    return 3 * (ema1 - ema2) + ema3

def tilson_t3(series: pd.Series, length: int, factor: float) -> pd.Series:
    def gd(s: pd.Series, l: int, f: float) -> pd.Series:
        e1 = ema(s, l)
        e2 = ema(e1, l)
        return e1 * (1 + f) - e2 * f
    if length < 1 or factor <= 0:
        return pd.Series(np.nan, index=series.index)
    t3 = gd(gd(gd(series, length, factor), length, factor), length, factor)
    return t3

MA_TYPES = {
    1: sma,
    2: ema,
    3: wma,
    4: hullma,
    5: vwma,
    6: rma,
    7: tema,
    8: tilson_t3
}

# ────────────────────────────────────────────────
# Backtest Logic
# ────────────────────────────────────────────────
def run_backtest(
    df: pd.DataFrame,
    len1: int, atype1: int, factor_t3_1: float,
    use_ma2: bool, len2: int, atype2: int, factor_t3_2: float,
    use_price_cross_ma1: bool, use_ma_cross: bool,
    initial_capital: float, commission_pct: float, slippage_points: float
) -> dict:

    if df.empty:
        return {"error": "Empty DataFrame"}

    src = df['Close']
    vol = df.get('Volume', pd.Series(1.0, index=df.index))

    # MA1
    if atype1 == 8:
        ma1 = tilson_t3(src, len1, factor_t3_1)
    elif atype1 == 5:
        ma1 = vwma(src, vol, len1)
    else:
        ma1 = MA_TYPES[atype1](src, len1)

    # MA2 (optional)
    ma2 = None
    if use_ma2:
        if atype2 == 8:
            ma2 = tilson_t3(src, len2, factor_t3_2)
        elif atype2 == 5:
            ma2 = vwma(src, vol, len2)
        else:
            ma2 = MA_TYPES[atype2](src, len2)

    df = df.copy()
    df['MA1'] = ma1
    if use_ma2 and ma2 is not None:
        df['MA2'] = ma2

    # Signals
    cr_up    = (df['Open'].shift(1) < df['MA1'].shift(1)) & (df['Close'] > df['MA1'])
    cr_down  = (df['Open'].shift(1) > df['MA1'].shift(1)) & (df['Close'] < df['MA1'])

    ma_cross_up = pd.Series(False, index=df.index)
    ma_cross_down = pd.Series(False, index=df.index)
    if use_ma2 and 'MA2' in df.columns:
        ma_cross_up   = (df['MA1'].shift(1) < df['MA2'].shift(1)) & (df['MA1'] > df['MA2'])
        ma_cross_down = (df['MA1'].shift(1) > df['MA2'].shift(1)) & (df['MA1'] < df['MA2'])

    long_signal  = (use_price_cross_ma1 & cr_up)   | (use_ma_cross & ma_cross_up)
    short_signal = (use_price_cross_ma1 & cr_down) | (use_ma_cross & ma_cross_down)

    # Simulation
    position = 0  # 1 = long, -1 = short, 0 = flat
    entry_price = 0.0
    equity_curve = [initial_capital]
    trades = []

    for i in range(1, len(df)):
        price = df['Close'].iloc[i]
        cost = (commission_pct / 100) * price + slippage_points

        if position == 0:
            if long_signal.iloc[i]:
                position = 1
                entry_price = price + cost
                trades.append({'Date': df.index[i], 'Action': 'BUY', 'Price': entry_price})
            elif short_signal.iloc[i]:
                position = -1
                entry_price = price - cost
                trades.append({'Date': df.index[i], 'Action': 'SELL SHORT', 'Price': entry_price})

        elif position == 1:
            if short_signal.iloc[i]:
                # Close long
                pnl = (price - entry_price - cost) / entry_price
                initial_capital *= (1 + pnl)
                trades.append({'Date': df.index[i], 'Action': 'SELL (close long)', 'Price': price, 'PnL %': pnl*100})
                # Open short
                position = -1
                entry_price = price - cost
                trades.append({'Date': df.index[i], 'Action': 'SELL SHORT', 'Price': entry_price})

        elif position == -1:
            if long_signal.iloc[i]:
                # Close short
                pnl = (entry_price - price - cost) / entry_price
                initial_capital *= (1 + pnl)
                trades.append({'Date': df.index[i], 'Action': 'BUY TO COVER', 'Price': price, 'PnL %': pnl*100})
                # Open long
                position = 1
                entry_price = price + cost
                trades.append({'Date': df.index[i], 'Action': 'BUY', 'Price': entry_price})

        equity_curve.append(initial_capital)

    # Close open position at end
    if position != 0:
        price = df['Close'].iloc[-1]
        cost = (commission_pct / 100) * price + slippage_points
        if position == 1:
            pnl = (price - entry_price - cost) / entry_price
            trades.append({'Date': df.index[-1], 'Action': 'SELL (final close)', 'Price': price, 'PnL %': pnl*100})
        else:
            pnl = (entry_price - price - cost) / entry_price
            trades.append({'Date': df.index[-1], 'Action': 'BUY TO COVER (final)', 'Price': price, 'PnL %': pnl*100})
        initial_capital *= (1 + pnl)

    # Assign equity (length now matches df)
    df['Equity'] = equity_curve

    # Metrics
    closed_trades = [t for t in trades if 'PnL %' in t]
    num_trades = len(closed_trades)
    win_rate = len([t for t in closed_trades if t['PnL %'] > 0]) / num_trades * 100 if num_trades > 0 else 0
    total_return_pct = (initial_capital / equity_curve[0] - 1) * 100

    metrics = {
        "Final Equity": round(initial_capital, 2),
        "Total Return (%)": round(total_return_pct, 2),
        "Number of Trades": num_trades,
        "Win Rate (%)": round(win_rate, 2),
        "Max Equity": round(max(equity_curve), 2),
        "Min Equity": round(min(equity_curve), 2),
    }

    return {
        "df": df,
        "trades": pd.DataFrame(trades),
        "metrics": metrics,
        "equity_curve": equity_curve
    }

# ────────────────────────────────────────────────
# Streamlit App
# ────────────────────────────────────────────────
st.title("MA Crossover Backtester (Hourly Candles)")

st.markdown("""
Upload CSV with columns: **Date**, **Open**, **High**, **Low**, **Close**  
( **Volume** is optional — will default to 1 if missing )

Date example: `Mon Feb 09 2026 09:15:00 GMT+0530 (India Standard Time)`
""")

uploaded_file = st.file_uploader("Upload your hourly CSV", type=["csv"])

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        raw_df['Date'] = raw_df['Date'].apply(parse_date)
        raw_df = raw_df.dropna(subset=['Date'])
        raw_df.set_index('Date', inplace=True)
        raw_df.sort_index(inplace=True)

        if not all(col in raw_df.columns for col in ['Open','High','Low','Close']):
            st.error("CSV must contain at least: Date, Open, High, Low, Close")
            st.stop()

        st.success(f"Loaded {len(raw_df)} candles from {raw_df.index.min()} to {raw_df.index.max()}")

        # ── Sidebar Controls ───────────────────────────────────────
        st.sidebar.header("Strategy Settings")

        len1 = st.sidebar.slider("MA1 Length", 5, 200, 20)
        atype1_name = st.sidebar.selectbox("MA1 Type", 
            ["SMA","EMA","WMA","HullMA","VWMA","RMA","TEMA","Tilson T3"],
            index=0)
        atype1 = ["SMA","EMA","WMA","HullMA","VWMA","RMA","TEMA","Tilson T3"].index(atype1_name) + 1
        factor_t3_1 = st.sidebar.slider("T3 Factor MA1 (0.0–2.0)", 0.0, 2.0, 0.7, 0.1) if atype1 == 8 else 0.0

        use_ma2 = st.sidebar.checkbox("Use second MA", value=True)
        len2 = st.sidebar.slider("MA2 Length", 10, 300, 50) if use_ma2 else 20
        atype2_name = st.sidebar.selectbox("MA2 Type", 
            ["SMA","EMA","WMA","HullMA","VWMA","RMA","TEMA","Tilson T3"],
            index=0) if use_ma2 else "SMA"
        atype2 = ["SMA","EMA","WMA","HullMA","VWMA","RMA","TEMA","Tilson T3"].index(atype2_name) + 1
        factor_t3_2 = st.sidebar.slider("T3 Factor MA2 (0.0–2.0)", 0.0, 2.0, 0.7, 0.1) if use_ma2 and atype2 == 8 else 0.0

        use_price_cross = st.sidebar.checkbox("Trade on Price × MA1 cross", value=True)
        use_ma_cross = st.sidebar.checkbox("Trade on MA1 × MA2 cross (golden/death)", value=False) if use_ma2 else False

        st.sidebar.header("Trading Parameters")
        initial_capital = st.sidebar.number_input("Starting Capital", 1000.0, 1000000.0, 10000.0, step=1000.0)
        commission_pct = st.sidebar.number_input("Commission per trade (%)", 0.0, 1.0, 0.1, step=0.01)
        slippage_points = st.sidebar.number_input("Slippage (points)", 0.0, 50.0, 0.0, step=0.5)

        if st.sidebar.button("Run Backtest", type="primary"):
            with st.spinner("Running backtest..."):
                result = run_backtest(
                    raw_df, len1, atype1, factor_t3_1,
                    use_ma2, len2, atype2, factor_t3_2,
                    use_price_cross, use_ma_cross,
                    initial_capital, commission_pct, slippage_points
                )

                if "error" in result:
                    st.error(result["error"])
                else:
                    st.subheader("Performance Summary")
                    st.json(result["metrics"])

                    st.subheader("Equity Curve")
                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Scatter(
                        x=raw_df.index,
                        y=result["df"]['Equity'],
                        mode='lines',
                        name='Equity',
                        line=dict(color='royalblue')
                    ))
                    fig_eq.update_layout(height=450)
                    st.plotly_chart(fig_eq, use_container_width=True)

                    st.subheader("Price & Moving Averages")
                    fig_candle = go.Figure()
                    fig_candle.add_trace(go.Candlestick(
                        x=raw_df.index,
                        open=raw_df['Open'], high=raw_df['High'],
                        low=raw_df['Low'], close=raw_df['Close'],
                        name='OHLC'
                    ))
                    fig_candle.add_trace(go.Scatter(x=raw_df.index, y=result["df"]['MA1'], name='MA1', line=dict(color='orange')))
                    if use_ma2 and 'MA2' in result["df"].columns:
                        fig_candle.add_trace(go.Scatter(x=raw_df.index, y=result["df"]['MA2'], name='MA2', line=dict(color='purple')))
                    fig_candle.update_layout(height=550, xaxis_rangeslider_visible=True)
                    st.plotly_chart(fig_candle, use_container_width=True)

                    st.subheader("Trade Log")
                    if not result["trades"].empty:
                        st.dataframe(result["trades"])
                    else:
                        st.info("No trades triggered with current settings.")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.exception(e)
