import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# --- SET UP PAGE ---
st.set_page_config(page_title="Options Parity Monitor", layout="wide")
st.title(" 📊 Real-Time Put-Call Parity Monitor")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", value="SPY").upper()
risk_free_rate = st.sidebar.number_input("Risk Free Rate (Decimal)", value=0.044, step=0.001)

# --- THE SAME LOGIC YOU WROTE YESTERDAY ---
# (I wrapped it in a function for caching so it doesn't reload every click)
@st.cache_data
def get_data(ticker_symbol):
    try:
        tk = yf.Ticker(ticker_symbol)
        exps = tk.options
        if not exps: return None, None
        
        # Default to a near-term expiration
        expiration = exps[2] if len(exps) > 2 else exps[0]
        
        # Fetch Chain
        opt = tk.option_chain(expiration)
        calls = opt.calls[['strike', 'lastPrice']].rename(columns={'lastPrice': 'call_price'})
        puts = opt.puts[['strike', 'lastPrice']].rename(columns={'lastPrice': 'put_price'})
        
        # Merge
        chain = pd.merge(calls, puts, on='strike')
        
        # Get Spot
        hist = tk.history(period="1d")
        if hist.empty: return None, None
        current_spot = hist['Close'].iloc[0]
        
        # Get Time (T)
        exp_dt = datetime.strptime(expiration, "%Y-%m-%d")
        days = (exp_dt - datetime.now()).days
        T = max(days / 365.0, 0.001)
        
        return chain, current_spot, T, expiration
    except Exception as e:
        return None, None

# --- MAIN APP LOGIC ---
if st.button("Run Analysis"):
    with st.spinner('Fetching Market Data...'):
        chain, S, T, exp_date = get_data(ticker)
    
    if chain is not None:
        # DO THE MATH (Same as before)
        chain['pv_strike'] = chain['strike'] * np.exp(-risk_free_rate * T)
        chain['call_plus_cash'] = chain['call_price'] + chain['pv_strike']
        chain['put_plus_stock'] = chain['put_price'] + S
        chain['deviation'] = chain['call_plus_cash'] - chain['put_plus_stock']
        
        # --- DISPLAY METRICS ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Spot Price", f"${S:.2f}")
        col2.metric("Expiration", exp_date)
        col3.metric("Time (Years)", f"{T:.4f}")
        
        # --- DISPLAY CHART ---
        st.subheader("Parity Deviation Structure")
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ['red' if v < 0 else 'green' for v in chain['deviation']]
        ax.bar(chain['strike'], chain['deviation'], color=colors)
        ax.axhline(0, color='black', linewidth=1)
        ax.set_xlabel("Strike Price")
        ax.set_ylabel("Arbitrage Deviation ($)")
        st.pyplot(fig)
        
        # --- DISPLAY DATA ---
        with st.expander("View Raw Data"):
            st.dataframe(chain)
            
    else:
        st.error("Could not fetch data. Check ticker symbol.")