import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# --- CONFIGURATION ---
TICKER_SYMBOL = "SPY"
RISK_FREE_RATE = 0.044  # Approx 4.4% yield on T-Bills
EXPIRATION_INDEX = 2    # 0 is the closest expiration, 1 is next, etc.

def fetch_option_data(ticker_symbol):
    """
    Fetches the option chain for a specific expiration date.
    """
    tk = yf.Ticker(ticker_symbol)
    
    # Get list of expiration dates
    exps = tk.options
    if not exps:
        print(f"No options found for {ticker_symbol}")
        return None
    
    # Pick an expiration date (e.g., the 3rd one available)
    expiration_date = exps[EXPIRATION_INDEX]
    print(f"Fetching data for Expiration: {expiration_date}...")
    
    # Get the chain
    opt = tk.option_chain(expiration_date)
    calls = opt.calls
    puts = opt.puts
    
    # --- CLEANING DATA ---
    # We only need Strike, Last Price, and Bid/Ask (using Last Price for MVP simplicity)
    calls = calls[['strike', 'lastPrice', 'volume']].rename(columns={'lastPrice': 'call_price'})
    puts = puts[['strike', 'lastPrice', 'volume']].rename(columns={'lastPrice': 'put_price'})
    
    # Merge Calls and Puts where the Strike is the same
    # This aligns the rows so we can do the math
    chain = pd.merge(calls, puts, on='strike')
    
    # Add Current Spot Price of the Stock
    # (Using a quick hack: average of high/low for today, or last close)
    hist = tk.history(period="1d")
    current_spot = hist['Close'].iloc[0]
    
    # Calculate Time to Expiration (T) in years
    exp_dt = datetime.strptime(expiration_date, "%Y-%m-%d")
    today = datetime.now()
    days_to_exp = (exp_dt - today).days
    T = max(days_to_exp / 365.0, 0.001) # Avoid division by zero
    
    return chain, current_spot, T, expiration_date

def calculate_parity(chain, S, T, r):
    """
    Put-Call Parity Formula: C + K * e^(-rT) = P + S
    We want to find the deviation: (C + K * e^(-rT)) - (P + S)
    """
    import numpy as np
    
    # Left Side: Call + Present Value of Strike (K)
    # K * e^(-rT)
    chain['pv_strike'] = chain['strike'] * np.exp(-r * T)
    chain['call_plus_cash'] = chain['call_price'] + chain['pv_strike']
    
    # Right Side: Put + Spot Price (S)
    chain['put_plus_stock'] = chain['put_price'] + S
    
    # The Arbitrage Deviation
    chain['deviation'] = chain['call_plus_cash'] - chain['put_plus_stock']
    
    return chain

def plot_parity(chain, expiration_date):
    """
    Visualizes the deviation across different strike prices.
    """
    plt.figure(figsize=(10, 6))
    
    plt.bar(chain['strike'], chain['deviation'], color='purple', width=2.0)
    
    plt.axhline(0, color='black', linewidth=1)
    plt.title(f"Put-Call Parity Deviation for {TICKER_SYMBOL} (Exp: {expiration_date})")
    plt.xlabel("Strike Price")
    plt.ylabel("Deviation ($)")
    plt.grid(True, alpha=0.3)
    
    # Save it so we can look at it
    filename = "parity_chart.png"
    plt.savefig(filename)
    print(f"Chart saved as {filename}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Get Data
    data = fetch_option_data(TICKER_SYMBOL)
    
    if data:
        chain, S, T, exp_date = data
        print(f"Spot Price (S): ${S:.2f}")
        print(f"Time to Exp (T): {T:.4f} years")
        
        # 2. Do Math
        df = calculate_parity(chain, S, T, RISK_FREE_RATE)
        
        # 3. Show Results (First 5 rows)
        print("\n--- DATA PREVIEW ---")
        print(df[['strike', 'call_price', 'put_price', 'deviation']].head())
        
        # 4. Make Chart
        plot_parity(df, exp_date)