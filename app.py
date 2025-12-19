import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.stats import norm

# --- CONFIGURATION ---
st.set_page_config(page_title="Pro-Quant Risk Dashboard", layout="wide", page_icon="🛡️")

# --- MATH ENGINE: BLACK-SCHOLES GREEKS ---
def calculate_greeks(S, K, T, r, sigma, option_type):
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            delta = norm.cdf(d1)
            theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
        else:
            delta = norm.cdf(d1) - 1
            theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100.0
        return delta, gamma, theta, vega
    except:
        return 0, 0, 0, 0

# --- STEP 1: FETCH AVAILABLE EXPIRATIONS (Lightweight) ---
@st.cache_data
def get_expirations(ticker_symbol):
    try:
        tk = yf.Ticker(ticker_symbol)
        exps = tk.options
        return exps
    except:
        return []

# --- STEP 2: FETCH SPECIFIC CHAIN (Heavyweight) ---
@st.cache_data
def get_chain(ticker_symbol, selected_date):
    try:
        tk = yf.Ticker(ticker_symbol)
        opt = tk.option_chain(selected_date)
        
        calls = opt.calls[['strike', 'lastPrice', 'impliedVolatility']].rename(columns={'lastPrice': 'call_price', 'impliedVolatility': 'call_iv'})
        puts = opt.puts[['strike', 'lastPrice', 'impliedVolatility']].rename(columns={'lastPrice': 'put_price', 'impliedVolatility': 'put_iv'})
        
        chain = pd.merge(calls, puts, on='strike')
        
        hist = tk.history(period="1d")
        if hist.empty: return None, None
        current_spot = hist['Close'].iloc[0]
        
        return chain, current_spot
    except:
        return None, None

# =========================================================
# 1. SIDEBAR - GLOBAL SETTINGS
# =========================================================
st.sidebar.header("Configuration")
page = st.sidebar.radio("Select Tool", ["Parity Monitor", "Iron Condor Builder"])
st.sidebar.markdown("---")

# A. Ticker Input
raw_ticker = st.sidebar.text_input("Ticker Symbol", value="SPY").upper()

# Auto-convert common indices to Yahoo format
if raw_ticker in ["SPX", "VIX", "NDX", "RUT"]:
    ticker = f"^{raw_ticker}"
else:
    ticker = raw_ticker

# Display the "Official" ticker being used
if ticker != raw_ticker:
    st.sidebar.caption(f"Indices require '^'. Using: {ticker}")

# B. Dynamic Expiration Selector
# We fetch the dates FIRST, so we can populate the dropdown
exps = get_expirations(ticker)
if not exps:
    st.error("Invalid Ticker or No Options Found")
    st.stop()

# Default to the 3rd expiration if available (usually a monthly), otherwise the 1st
default_idx = 2 if len(exps) > 2 else 0
selected_exp = st.sidebar.selectbox("Expiration Date", exps, index=default_idx)

risk_free_rate = st.sidebar.number_input("Risk Free Rate (Decimal)", value=0.044, step=0.001)

# =========================================================
# 2. DATA FETCHING LAYER
# =========================================================
# Now we fetch the chain based on the USER'S choice from the sidebar
if 'chain' not in st.session_state: st.session_state.chain = None
if 'spot' not in st.session_state: st.session_state.spot = None
if 'last_ticker' not in st.session_state: st.session_state.last_ticker = None
if 'last_exp' not in st.session_state: st.session_state.last_exp = None

# Logic: Refetch if Ticker changed OR Expiration changed
if (st.session_state.chain is None or 
    st.session_state.last_ticker != ticker or 
    st.session_state.last_exp != selected_exp):
    
    with st.spinner('Fetching Chain Data...'):
        chain, S = get_chain(ticker, selected_exp)
        if chain is not None:
            st.session_state.chain = chain
            st.session_state.spot = S
            st.session_state.last_ticker = ticker
            st.session_state.last_exp = selected_exp
            st.session_state.fetch_time = datetime.now().strftime("%H:%M:%S")

chain = st.session_state.chain
S = st.session_state.spot
fetch_time = st.session_state.fetch_time

if chain is None:
    st.error("Could not fetch data.")
    st.stop()

# =========================================================
# 3. SIDEBAR - STRATEGY CONTROLS (Bottom)
# =========================================================
if page == "Iron Condor Builder":
    st.sidebar.markdown("---")
    st.sidebar.header("Strategy Settings")
    
    outlook = st.sidebar.radio("Market Outlook", ["Neutral", "Volatile"], index=0)
    is_short_condor = "Neutral" in outlook
    
    # Strike helpers
    available_strikes = sorted(chain['strike'].unique())
    try:
        spot_idx = min(range(len(available_strikes)), key=lambda i: abs(available_strikes[i]-S))
        lower_idx = max(0, spot_idx - 5)
        upper_idx = min(len(available_strikes)-1, spot_idx + 5)
    except:
        lower_idx, upper_idx = 0, len(available_strikes)-1

    st.sidebar.subheader("Put Wing")
    put_strikes = st.sidebar.select_slider("Low / High Put", options=available_strikes, value=(available_strikes[max(0, lower_idx-2)], available_strikes[lower_idx]))
    
    st.sidebar.subheader("Call Wing")
    call_strikes = st.sidebar.select_slider("Low / High Call", options=available_strikes, value=(available_strikes[upper_idx], available_strikes[min(len(available_strikes)-1, upper_idx+2)]))
    
    outer_put_k, inner_put_k = put_strikes
    inner_call_k, outer_call_k = call_strikes

# =========================================================
# 4. MAIN PAGE DISPLAY
# =========================================================
st.title(f"🛡️ Risk Terminal: {ticker}")
st.caption(f"Expiration: {selected_exp} | Data: {fetch_time}")

# Time to Expiration Calculation (Uses the selected date!)
exp_dt = datetime.strptime(selected_exp, "%Y-%m-%d")
days = (exp_dt - datetime.now()).days
T = max(days / 365.0, 0.001)

if page == "Parity Monitor":
    st.header("Put-Call Parity Analysis")
    
    chain['pv_strike'] = chain['strike'] * np.exp(-risk_free_rate * T)
    chain['call_plus_cash'] = chain['call_price'] + chain['pv_strike']
    chain['put_plus_stock'] = chain['put_price'] + S
    chain['deviation'] = chain['call_plus_cash'] - chain['put_plus_stock']
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Spot Price", f"${S:.2f}")
    c2.metric("Days to Exp", f"{days} days")
    c3.metric("Avg Deviation", f"${chain['deviation'].mean():.2f}")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['red' if v < 0 else 'green' for v in chain['deviation']]
    ax.bar(chain['strike'], chain['deviation'], color=colors)
    ax.axhline(0, color='black', linewidth=1)
    st.pyplot(fig)

elif page == "Iron Condor Builder":
    st.header("Strategy Visualization")
    
    # 1. Get Rows
    row_op = chain.loc[chain['strike'] == outer_put_k].iloc[0]
    row_ip = chain.loc[chain['strike'] == inner_put_k].iloc[0]
    row_ic = chain.loc[chain['strike'] == inner_call_k].iloc[0]
    row_oc = chain.loc[chain['strike'] == outer_call_k].iloc[0]

    # 2. Greeks
    g_op = calculate_greeks(S, outer_put_k, T, risk_free_rate, row_op['put_iv'], 'put')
    g_ip = calculate_greeks(S, inner_put_k, T, risk_free_rate, row_ip['put_iv'], 'put')
    g_ic = calculate_greeks(S, inner_call_k, T, risk_free_rate, row_ic['call_iv'], 'call')
    g_oc = calculate_greeks(S, outer_call_k, T, risk_free_rate, row_oc['call_iv'], 'call')
    
    if is_short_condor:
        mult_inner, mult_outer = -1, 1
    else:
        mult_inner, mult_outer = 1, -1

    net_delta = (g_ip[0]*mult_inner) + (g_ic[0]*mult_inner) + (g_op[0]*mult_outer) + (g_oc[0]*mult_outer)
    net_theta = (g_ip[2]*mult_inner) + (g_ic[2]*mult_inner) + (g_op[2]*mult_outer) + (g_oc[2]*mult_outer)
    net_vega  = (g_ip[3]*mult_inner) + (g_ic[3]*mult_inner) + (g_op[3]*mult_outer) + (g_oc[3]*mult_outer)

    # 3. P/L Math
    if is_short_condor:
        net_prem = (row_ip['put_price'] + row_ic['call_price']) - (row_op['put_price'] + row_oc['call_price'])
        max_risk = max(inner_put_k - outer_put_k, outer_call_k - inner_call_k) - net_prem
    else:
        net_prem = (row_ip['put_price'] + row_ic['call_price']) - (row_op['put_price'] + row_oc['call_price'])
        max_risk = net_prem # For long, risk is the cost
    
    # 4. METRICS ROW 1: P/L
    st.subheader("💰 Profit Profile")
    m1, m2, m3, m4 = st.columns(4)
    if is_short_condor:
        m1.metric("Net Credit", f"${net_prem:.2f}", help="Cash received upfront. This is your maximum possible profit.")
        m2.metric("Max Risk", f"${max_risk:.2f}", help="Maximum amount you can lose if the stock moves significantly past your wings.")
        m3.metric("Break-Even Low", f"${inner_put_k - net_prem:.2f}", help="Stock price where you start losing money on the downside.")
        m4.metric("Break-Even High", f"${inner_call_k + net_prem:.2f}", help="Stock price where you start losing money on the upside.")
    else:
        m1.metric("Cost to Enter", f"${net_prem:.2f}", help="Cash paid upfront. This is your maximum possible loss.")
        m2.metric("Max Profit", f"${max_risk:.2f}", help="Maximum amount you can earn if the stock moves significantly past your wings.")
        m3.metric("Break-Even Low", f"${inner_put_k - net_prem:.2f}", help="Stock price where you start making money on the downside.")
        m4.metric("Break-Even High", f"${inner_call_k + net_prem:.2f}", help="Stock price where you start making money on the upside.")

    # METRICS ROW 2: RISK (THE GREEKS)
    st.subheader("⚡ Risk Profile (Greeks)")
    g1, g2, g3 = st.columns(3)
    
    g1.metric(
        "Net Delta (Direction)", 
        f"{net_delta:.3f}", 
        help="Sensitivity to Price.\n\n"
             "• Positive (+): You make money if stock goes UP.\n"
             "• Negative (-): You make money if stock goes DOWN.\n"
             "• Near Zero: You are 'Delta Neutral' (Direction doesn't matter)."
    )
    
    g2.metric(
        "Net Theta (Time Decay)", 
        f"${net_theta:.2f}", 
        help="Sensitivity to Time.\n\n"
             "• Positive (+): You EARN cash every day that passes (Time is your friend).\n"
             "• Negative (-): You LOSE cash every day that passes (Time is your enemy)."
    )
    
    g3.metric(
        "Net Vega (Volatility)", 
        f"${net_vega:.2f}", 
        help="Sensitivity to Volatility (Panic).\n\n"
             "• Positive (+): You profit if Implied Volatility RISES (Panic).\n"
             "• Negative (-): You profit if Implied Volatility FALLS (Calm)."
    )

    # 5. Chart
    st.subheader("Payoff")
    plot_min = outer_put_k * 0.95
    plot_max = outer_call_k * 1.05
    prices = np.linspace(plot_min, plot_max, 100)
    payoffs = []
    for p in prices:
        if is_short_condor:
            val = (net_prem + max(outer_put_k - p, 0) - max(inner_put_k - p, 0) - max(p - inner_call_k, 0) + max(p - outer_call_k, 0))
        else:
            val = (-net_prem - max(outer_put_k - p, 0) + max(inner_put_k - p, 0) + max(p - inner_call_k, 0) - max(p - outer_call_k, 0))
        payoffs.append(val)
        
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(prices, payoffs, color='blue')
    ax2.fill_between(prices, payoffs, 0, where=(np.array(payoffs) > 0), color='green', alpha=0.3)
    ax2.fill_between(prices, payoffs, 0, where=(np.array(payoffs) < 0), color='red', alpha=0.3)
    ax2.axhline(0, color='black')
    st.pyplot(fig2)