# 🧠 Source Code

This folder contains all reusable core logic and setup utilities used across the project’s notebooks and future backtesting modules.

The code is structured to separate **data loading & configuration** from **financial analysis logic**, enabling clean, modular, and maintainable workflows.

---

## 📁 Files

### `analysis_functions.py`
Contains all core financial analysis functions used across the project, including:
- Balance sheet metrics
- Capital structure analysis
- Profitability & return calculations
- Valuation metrics (EV/EBITDA, PE, CRV, etc.)
- Risk-reward and probability-adjusted models

All functions are designed to be:
- Pure (no printing or plotting),
- Reusable,
- Notebook-agnostic,
- Suitable for future backtesting modules.

---

### `setup.py`
Central setup module responsible for:
- Importing core libraries (`pandas`, `numpy`, `matplotlib`, `yfinance`),
- Defining global financial constants and parameters (e.g. tax rate, beta, market return),
- Loading and cleaning raw financial data from `data/raw/`,
- Fetching live market data via Yahoo Finance,
- Creating standardized data structures used across all notebooks.

It exposes preconfigured objects such as:
- `df` – combined quarterly dataset  
- `annual_avg` – annualized financial metrics  
- `info` – Yahoo Finance ticker object  
- `balance_sheet` – latest balance sheet data  
- `price`, `market_cap`, `shares_outstanding` – current market values  

This allows notebooks to start with:

```python
from src.setup import *
