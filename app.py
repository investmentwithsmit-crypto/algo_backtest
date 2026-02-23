import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from typing import Callable

# Function to parse the date format provided
def parse_date(date_str: str) -> datetime.datetime:
    # Example: "Mon Feb 09 2026 09:15:00 GMT+0530 (India Standard Time)"
    # We need to handle the timezone part
    try:
        # Split off the timezone name if present
        if '(' in date_str:
            date_str = date_str.split(' (')[0]
        dt = datetime.datetime.strptime(date_str, "%a %b %d %Y %H:%M:%S %Z%z")
        return dt
    except ValueError:
        st.error(f"Invalid date format: {date_str}")
        return None

# MA calculation functions (adapted from Pine Script)
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
    t3 = gd(gd(gd(series, length, factor), length, factor), length, factor)
    return t3

# Dictionary to select MA function
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

# Backtest function
def run_backtest(df: pd.DataFrame, len1: int, atype1: int, factor_t3_1: float,
                 use_ma2: bool, len2: int, atype2: int, factor_t3_2: float,
                 use_price_cross_ma1: bool, use_ma_cross: bool,
                 initial_capital: float, commission: float, slippage: float) -> dict:
    
    src = df['Close']
    vol = df.get('Volume', pd.Series(1, index=df.index))  # Default volume if not present
    
    # Calculate MA1
    if atype1 == 8:
        ma1_func: Callable = lambda s, l: tilson_t3(s, l, factor_t3_1)
    elif atype1 == 5:
        ma1_func = lambda s, l: vwma(s, vol, l)
    else:
        ma1_func = MA_TYPES[atype1]
    ma1 = ma1_func(src, len1)
    
    # Calculate MA2 if used
    if use_ma2:
        if atype2 == 8:
            ma2_func: Callable = lambda s, l: tilson_t3(s, l, factor_t3_2)
        elif atype2 == 5:
            ma2_func = lambda s, l: vwma(s, vol, l)
        else:
            ma2_func = MA_TYPES[atype2]
        ma2 = ma2_func(src, len2)
    else:
        ma2 = None
    
    # Signals
    df['MA1'] = ma1
    if use_ma2:
        df['MA2'] = ma2
    
    # Price cross MA1
    cr_up = (df['Open'].shift(1) < ma1.shift(1)) & (df['Close'] > ma1)
    cr_down = (df['Open'].shift(1) > ma1.shift(1)) & (df['Close'] < ma1)
    
    # MA cross
    if use_ma2:
        ma_cross_up = (ma1.shift(1) < ma2.shift(1)) & (ma1 > ma2)
        ma_cross_down = (ma1.shift(1) > ma2.shift(1)) & (ma1 < ma2)
    else:
        ma_cross_up = pd.Series(False, index=df.index)
        ma_cross_down = pd.Series(False, index=df.index)
    
    # Combine signals based on user choice
    long_entry = (use_price_cross_ma1 & cr_up) | (use_ma_cross & ma_cross_up)
    short_entry = (use_price_cross_ma1 & cr_down) | (use_ma_cross & ma_cross_down)
    
    # Simulate trades
    position = 0  # 1: long, -1: short, 0: flat
    entry_price = 0
    equity = [initial_capital]
    trades = []
    
    for i in range(1, len(df)):
        current_price = df['Close'].iloc[i]
        entry_cost = commission / 100 * current_price + slippage
        
        if position == 0:
            if long_entry.iloc[i]:
                position = 1
                entry_price = current_price + entry_cost
                trades.append({'Date': df.index[i], 'Type': 'Buy', 'Price': entry_price})
            elif short_entry.iloc[i]:
                position = -1
                entry_price = current_price - entry_cost
                trades.append({'Date': df.index[i], 'Type': 'Sell Short', 'Price': entry_price})
        
        elif position == 1:
            if short_entry.iloc[i]:  # Close long and open short
                profit = (current_price - entry_price - entry_cost) / entry_price
                initial_capital *= (1 + profit)
                trades.append({'Date': df.index[i], 'Type': 'Sell', 'Price': current_price, 'Profit': profit})
                # Open short
                position = -1
                entry_price = current_price - entry_cost
                trades.append({'Date': df.index[i], 'Type': 'Sell Short', 'Price': entry_price})
            elif long_entry.iloc[i]:  # Ignore if already long
                pass
        
        elif position == -1:
            if long_entry.iloc[i]:  # Close short and open long
                profit = (entry_price - current_price - entry_cost) / entry_price
                initial_capital *= (1 + profit)
                trades.append({'Date': df.index[i], 'Type': 'Buy to Cover', 'Price': current_price, 'Profit': profit})
                # Open long
                position = 1
                entry_price = current_price + entry_cost
                trades.append({'Date': df.index[i], 'Type': 'Buy', 'Price': entry_price})
            elif short_entry.iloc[i]:  # Ignore if already short
                pass
        
        equity.append(initial_capital)
    
    # Close any open position at end
    if position != 0:
        current_price = df['Close'].iloc[-1]
        entry_cost = commission / 100 * current_price + slippage
        if position == 1:
            profit = (current_price - entry_price - entry_cost) / entry_price
            trades.append({'Date': df.index[-1], 'Type': 'Sell', 'Price': current_price, 'Profit': profit})
        else:
            profit = (entry_price - current_price - entry_cost) / entry_price
            trades.append({'Date': df.index[-1], 'Type': 'Buy to Cover', 'Price': current_price, 'Profit': profit})
        initial_capital *= (1 + profit)
    
    df['Equity'] = equity + [equity[-1]]  # Pad last equity
    
    # Metrics
    if trades:
        profits = [t.get('Profit', 0) for t in trades if 'Profit' in t]
        win_rate = len([p for p in profits if p > 0]) / len(profits) * 100 if profits else 0
        total_return = (initial_capital - equity[0]) / equity[0] * 100
    else:
        win_rate = 0
        total_return = 0
    
    return {
        'df': df,
        'trades': pd.DataFrame(trades),
        'metrics': {
            'Total Return (%)': total_return,
            'Win Rate (%)': win_rate,
            'Number of Trades': len([t for t in trades if 'Profit' in t]),
            'Final Equity': initial_capital
        }
    }

# Streamlit App
st.title("Hourly Candle Backtest Tool")
st.markdown("""
Upload your hourly candle data in CSV format with columns: Date, Open, High, Low, Close (Volume optional).
Date format: e.g., "Mon Feb 09 2026 09:15:00 GMT+0530 (India Standard Time)"
""")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # Parse dates
    df['Date'] = df['Date'].apply(parse_date)
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    # Inputs
    st.sidebar.header("Strategy Parameters")
    len1 = st.sidebar.slider("MA1 Length", 1, 200, 20)
    atype1 = st.sidebar.selectbox("MA1 Type", options=list(range(1,9)), format_func=lambda x: ["SMA", "EMA", "WMA", "HullMA", "VWMA", "RMA", "TEMA", "Tilson T3"][x-1], index=0)
    factor_t3_1 = st.sidebar.slider("Tilson T3 Factor MA1 (x0.1)", 0, 20, 7) / 10.0 if atype1 == 8 else 0
    
    use_ma2 = st.sidebar.checkbox("Use 2nd MA", value=True)
    len2 = st.sidebar.slider("MA2 Length", 1, 200, 50) if use_ma2 else 0
    atype2 = st.sidebar.selectbox("MA2 Type", options=list(range(1,9)), format_func=lambda x: ["SMA", "EMA", "WMA", "HullMA", "VWMA", "RMA", "TEMA", "Tilson T3"][x-1], index=0) if use_ma2 else 1
    factor_t3_2 = st.sidebar.slider("Tilson T3 Factor MA2 (x0.1)", 0, 20, 7) / 10.0 if use_ma2 and atype2 == 8 else 0
    
    use_price_cross_ma1 = st.sidebar.checkbox("Trade on Price Cross MA1", value=True)
    use_ma_cross = st.sidebar.checkbox("Trade on MA1 Cross MA2", value=False) if use_ma2 else False
    
    initial_capital = st.sidebar.number_input("Initial Capital", value=10000.0)
    commission = st.sidebar.number_input("Commission (%)", value=0.1)
    slippage = st.sidebar.number_input("Slippage (points)", value=0.0)
    
    if st.sidebar.button("Run Backtest"):
        results = run_backtest(df.copy(), len1, atype1, factor_t3_1, use_ma2, len2, atype2, factor_t3_2,
                               use_price_cross_ma1, use_ma_cross, initial_capital, commission, slippage)
        
        st.header("Backtest Results")
        st.subheader("Metrics")
        st.json(results['metrics'])
        
        st.subheader("Equity Curve")
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(x=results['df'].index, y=results['df']['Equity'], mode='lines', name='Equity'))
        st.plotly_chart(fig_equity)
        
        st.subheader("Price Chart with MAs and Signals")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=results['df'].index, open=results['df']['Open'], high=results['df']['High'],
                                     low=results['df']['Low'], close=results['df']['Close'], name='Candles'))
        fig.add_trace(go.Scatter(x=results['df'].index, y=results['df']['MA1'], mode='lines', name='MA1'))
        if use_ma2:
            fig.add_trace(go.Scatter(x=results['df'].index, y=results['df']['MA2'], mode='lines', name='MA2'))
        st.plotly_chart(fig)
        
        st.subheader("Trades")
        st.dataframe(results['trades'])
