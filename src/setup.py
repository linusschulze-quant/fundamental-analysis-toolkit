# =========================
# Imports
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# =========================
# Constants
# =========================
# Ticker
TICKER: str = "MSFT"
# Financial Parameters
BETA: float = 0.87
RISK_FREE_RATE: float = 0.04
MARKET_RETURN: float = 0.10
TAX_RATE: float = 19
CREDIT_SPREAD: float = 0.0005

# =========================
# Setup & Data Loading
# =========================

# --- Yahoo Finance Info ---
ticker: str = TICKER
info = yf.Ticker(ticker)

# --- Load CSV Data ---
df1 = pd.read_csv(f"data/raw/data(1).csv", parse_dates=["Date"])
df1.set_index("Date", inplace=True)

df2 = pd.read_csv(f"data/raw/data(2).csv", parse_dates=["Date"])
df2.set_index("Date", inplace=True)

# # Combine quarterly data from two CSVs based on the Date index
df = df1.join(df2, how="outer")
df["Year"] = df.index.year

# Columns to forward/backward fill
fill_columns = [
    f"{ticker}: Quarterly Long Term Debt",
    f"{ticker}: Quarterly Shareholders Equity",
    f"{ticker}: Market Cap",
    f"{ticker}: Quarterly Accounts Receivable",
    f"{ticker}: Shares Outstanding",
]

# Fill missing values in specified columns

fill_columns = [col for col in fill_columns if col in df.columns]
df[fill_columns] = df[fill_columns].ffill().bfill()

# =========================
# Annual Averages & Annualization
# =========================

# Calculate annual averages

annual_avg = df.groupby("Year").mean()

# Fill missing Free Cash Flow & Cash & Equivalents
for col in [f"{ticker}: Quarterly Free Cash Flow",
            f"{ticker}: Quarterly Cash & Cash Equivalents"]:
    annual_avg[col] = annual_avg[col].ffill().bfill()

# Optional: handle PE Ratio if present
pe_col = f"{ticker}: PE Ratio"
if pe_col in df.columns:
    df[pe_col] = pd.to_numeric(df[pe_col], errors="coerce").ffill().bfill()
    annual_avg[pe_col] = df.groupby("Year").mean()[pe_col]

# Annualize quarterly data
annual_avg[f"{ticker}: Annual Free Cash Flow"] = (
    annual_avg[f"{ticker}: Quarterly Free Cash Flow"] * 4
)
annual_avg[f"{ticker}: Annual EPS"] = (
    annual_avg[f"{ticker}: Quarterly EPS"] * 4
)

# =========================
# Market & Risk Parameters
# =========================

market_risk_premium: float = MARKET_RETURN - RISK_FREE_RATE
equity_risk_premium: float = BETA * market_risk_premium # Adjusted for stock's beta

# Latest shares outstanding
shares_outstanding: int = df[f"{ticker}: Shares Outstanding"].iloc[-1]

# Latest stock price and market cap
price: float = info.info.get("currentPrice", np.nan)
market_cap: float = info.info.get("marketCap", np.nan)

# Load balance sheet (check if available)
balance_sheet = info.balance_sheet
if balance_sheet is None:
    print("Warning: Balance sheet data not available.")
