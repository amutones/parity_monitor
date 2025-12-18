# SPY Put-Call Parity Monitor

## Project Overview
This tool is a Python-based quantitative monitor that ingests real-time option chain data for the S&P 500 ETF (SPY) to visualize pricing inefficiencies.

It calculates the theoretical **Put-Call Parity** deviation across all available strike prices for a given expiration, helping to identify potential arbitrage opportunities or liquidity gaps in the market.

## How It Works
The script utilizes the standard parity relationship for European options:
$$C + K e^{-rT} = P + S$$

* **Fetches Data:** Pulls live option chains and spot prices using the `yfinance` API.
* **Data Cleaning:** Merges Call and Put dataframes on Strike Price and filters for liquidity.
* **Math Engine:** Calculates the Present Value (PV) of the Strike and computes the deviation spread.
* **Visualization:** Generates a bar chart to highlight the "Smile" effect caused by liquidity premiums at OTM/ITM strikes.

## Key Findings (The "Smile" Effect)
In running this monitor against live SPY data, the visualization typically reveals a "Volatility Smile" or "Liquidity Smile" structure:
* **ATM Strikes:** Deviation is near-zero, indicating high market efficiency and tight spreads.
* **Deep OTM/ITM Strikes:** Deviation increases significantly. This is attributed to wider bid-ask spreads in illiquid strikes and the "American Option" early-exercise premium (which the standard European formula does not account for).

## Technical Stack
* **Python 3.10+**
* **Pandas:** For dataframe merging and vectorised calculations.
* **Matplotlib:** For data visualization.
* **yfinance:** For market data ingestion.

## How to Run Locally
1.  **Clone the repository**
    ```bash
    git clone [https://github.com/amutones/parity_monitor.git](https://github.com/amutones/parity_monitor.git)
    cd parity_monitor
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the monitor**
    ```bash
    python parity_monitor.py
    ```
    *Output:* The script will generate a `parity_chart.png` file in the root directory.

---
*Author: Anthony Urgena | Risk & Compliance Analyst*
