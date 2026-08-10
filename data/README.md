# Data

This directory contains the stored datasets used by the Fundamental Analysis Toolkit.

The data is separated into two layers:

```text
data/
├── raw/
└── snapshot/
```

This separation supports reproducible analysis while keeping historical series distinct from fixed financial-statement snapshots.

## Raw Data

`data/raw/` contains historical financial and valuation series used primarily for time-series and historical distribution analysis.

Current files:

```text
data(1).csv
data(2).csv
```

The historical data includes metrics such as:

* Free Cash Flow
* EPS
* Cash
* Long-Term Debt
* Stockholders' Equity
* Accounts Receivable
* Shares Outstanding
* Market Capitalization
* P/E
* EV/EBITDA

Historical observations begin in **2006 where available** and extend through **2025**.

These datasets originate from CSV exports from **Wall Street Numbers**.

## Snapshot Data

`data/snapshot/` contains fixed company, market, and financial-statement snapshots used for point-in-time and accounting-based analysis.

Included files:

```text
balance_sheet.csv
cash_flow.csv
company_profile.json
income_statement.csv
market_snapshot.csv
metadata.json
quarterly_balance_sheet.csv
quarterly_cash_flow.csv
quarterly_income_statement.csv
```

The current snapshot is configured for **Microsoft Corporation (MSFT)**.

Key dates:

| Data                           | Latest Stored Period |
| ------------------------------ | -------------------- |
| Market Data                    | December 26, 2025    |
| Quarterly Financial Statements | September 30, 2025   |
| Annual Financial Statements    | FY2025               |

The snapshot data was originally retrieved from **Yahoo Finance via `yfinance`** and is stored locally so the analytical notebooks do not require live API requests.

Additional information about the snapshot source and creation date is available in:

```text
snapshot/metadata.json
```

## Why Stored Data?

The project intentionally separates data acquisition from analysis.

Using fixed datasets ensures that:

* notebook outputs remain reproducible
* later revisions to external data do not silently change previous results
* API availability does not affect execution
* reporting periods can be documented explicitly
* historical and point-in-time inputs remain distinguishable

## Important Data Considerations

Historical raw-series definitions do not always match the financial-statement definitions used in the stored snapshots.

For example, historical cash or debt series may represent broader or differently classified financial positions than the corresponding Yahoo Finance statement rows.

For this reason:

* current accounting analysis primarily uses the stored financial-statement snapshots
* historical raw data is used mainly for historical and distribution analysis
* analyses that combine different periods or definitions state those limitations explicitly

## Usage

Data loading and preparation are centralized in:

```text
../src/setup.py
```

The notebooks access the prepared datasets through:

```python
from src.setup import *
```

For the full project methodology and notebook descriptions, see the [main README](../README.md).
