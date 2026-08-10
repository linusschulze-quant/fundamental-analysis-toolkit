# Analysis Modules

This directory contains the seven Jupyter notebooks that form the analytical core of the Fundamental Analysis Toolkit.

Each notebook focuses on a separate part of fundamental equity research. The analyses share the same prepared datasets from `src/setup.py`, but use annual, quarterly, trailing, historical, or point-in-time data depending on the metric being calculated.

---

## 1. [Balance Sheet Analysis](01_balance_sheet_analysis.ipynb)

The balance sheet notebook evaluates the company's latest quarterly financial position.

It calculates:

* Total Assets
* Total Debt
* Stockholders' Equity
* Cash and Cash Equivalents
* Long-Term Debt
* Net Debt
* Gearing Ratio
* Equity Ratio
* Debt-to-Assets Ratio

The analysis uses the latest available quarterly balance-sheet snapshot consistently for all calculations.

For the current MSFT dataset, the balance-sheet date is **September 30, 2025**.

---

## 2. [Capital Analysis](02_capital_analysis.ipynb)

This notebook analyses capital structure, financing costs, and capital efficiency.

It includes:

* Book Equity Ratio
* Debt-to-Assets Ratio
* Market-value WACC weights
* CAPM-based Cost of Equity
* After-tax Cost of Debt
* Weighted Average Cost of Capital
* Interest Coverage Ratio
* Revenue-to-Assets Ratio

WACC uses market capitalization for the equity weight and the latest quarterly Total Debt for the debt weight.

Revenue-to-Assets uses **FY2025 revenue** and average Total Assets from **FY2024 and FY2025**, avoiding a direct comparison between an annual flow and a single period-end asset balance.

---

## 3. [Discounted Cash Flow Analysis](03_dcf_analysis.ipynb)

The DCF notebook estimates intrinsic equity value using projected Free Cash Flow.

The model:

1. starts from trailing-twelve-month Free Cash Flow
2. projects Free Cash Flow over ten years
3. discounts projected cash flows using WACC
4. calculates terminal value using the Gordon Growth Model
5. adds cash and subtracts debt
6. converts the resulting equity value into value per share

### Current Model Assumptions

* Forecast FCF Growth: **5%**
* Terminal Growth: **3%**
* WACC: approximately **9.12%**
* Forecast Horizon: **10 years**

Using the stored MSFT data, the model produces an illustrative value of approximately **$201 per share**.

This value is a model output under the stated assumptions and should not be interpreted as a price forecast.

### Historical DCF

The notebook also contains an illustrative historical DCF section.

It applies the current growth and discount-rate assumptions to historical financial inputs. Because the historical raw cash and debt definitions can differ from the current financial-statement snapshots, this section is **not a true point-in-time backtest**.

---

## 4. [Profitability Analysis](04_profitability_analysis.ipynb)

The profitability notebook combines operating performance, capital efficiency, cash generation, growth, and valuation metrics.

### Margins

* Gross Margin
* Net Margin
* EBIT Margin
* EBITDA Margin

### Returns

* Return on Equity
* Return on Assets
* Simplified ROIC

ROE and ROA use average annual balance-sheet values rather than only year-end balances.

### Cash Flow and Operating Performance

* Free Cash Flow Yield
* Degree of Operating Leverage
* Historical Free Cash Flow Growth

### Growth and Valuation

* EPS (TTM)
* P/E
* Earnings Yield
* Average historical EPS Growth
* Trailing PEG Ratio

The notebook explicitly labels ROIC as **Simplified ROIC**, since invested capital is approximated using:

```text
Stockholders' Equity + Total Debt - Cash
```

---

## 5. [Company Information](05_company_info.ipynb)

This notebook combines stored company metadata with point-in-time market information and balance-sheet-derived valuation metrics.

It includes:

* Company Name
* Country
* Sector
* Industry
* Currency
* Website
* Share Price
* Market Capitalization
* Book Value per Share
* Price-to-Book Ratio
* Business Description

Book Value per Share uses both **Stockholders' Equity** and **Ordinary Shares Number** from the same quarterly reporting date: **September 30, 2025**.

The market price and market capitalization refer to **December 26, 2025**, and the different dates are explicitly shown in the output.

---

## 6. [Beneish M-Score](06_beneish_m_score.ipynb)

This notebook implements the eight-variable Beneish M-Score as a forensic accounting screening tool.

The model includes:

* **DSRI** — Days Sales in Receivables Index
* **GMI** — Gross Margin Index
* **AQI** — Asset Quality Index
* **SGI** — Sales Growth Index
* **DEPI** — Depreciation Index
* **SGAI** — SG&A Expense Index
* **LVGI** — Leverage Index
* **TATA** — Total Accruals to Total Assets

The calculation compares aligned annual financial statements for **FY2025 and FY2024**.

The current MSFT analysis produces an M-Score of approximately:

```text
-2.55
```

A score below the commonly used **-2.22** threshold represents a lower indication of earnings manipulation under the model.

The Beneish M-Score is treated as a **screening indicator rather than proof of accounting manipulation**.

---

## 7. [Historical Valuation & Risk/Reward Analysis](07_risk_reward_analysis.ipynb)

The final notebook analyses the historical distribution of Microsoft's:

* P/E Ratio
* EV/EBITDA Multiple

It combines a valuation reference system with broader statistical analysis.

### WC / Buy / Fair Value / Sell

WC / Buy / FV / Sell reference levels are based on **annual-average valuation multiples**.

This reduces the influence of isolated short-term extreme observations when defining the four reference levels.

The resulting multiples are translated into implied share prices using:

* TTM EPS for P/E
* FY2025 EBITDA
* latest stored Net Debt
* latest stored Shares Outstanding

The four levels are:

* **WC** — downside reference level
* **Buy** — lower historical valuation reference
* **FV** — midpoint between Buy and Sell
* **Sell** — upper historical valuation reference

These levels describe historical valuation ranges and are not forecasts.

### Historical Distribution Analysis

Quantiles, frequencies, and histograms use the **full stored historical valuation series** rather than annual averages.

The current dataset contains:

* **1,042 P/E observations**
* **1,042 EV/EBITDA observations**

The notebook calculates:

* Q10
* Q25
* Q50 Median
* Q75
* Q90
* Quantile-implied Share Prices
* Historical Frequency Above the Current Multiple
* Historical Frequency At or Below the Current Multiple
* Full 15-bin historical distributions

### Risk/Reward

Risk/reward compares the current share price with the WC and Sell implied prices.

The analysis reports:

* Potential Upside ("Chance")
* Potential Downside ("Risk")
* Risk/Reward Ratio

Historical frequencies are presented separately and are **not used as forecast probabilities or probability weights**.

This distinction prevents descriptive historical statistics from being interpreted as predictive probabilities.

---

## Data Periods

The notebooks intentionally use different datasets depending on the required metric.

| Data Type                            | Period                                     |
| ------------------------------------ | ------------------------------------------ |
| Market Price / Market Capitalization | December 26, 2025                          |
| Latest Quarterly Balance Sheet       | September 30, 2025                         |
| Latest Annual Financial Statements   | FY2025                                     |
| TTM Metrics                          | Latest available observations through 2025 |
| Historical Valuation Series          | 2006–2025 where available                  |

Whenever data from different reporting periods must be combined, the dates and methodological limitations are stated explicitly.

---

## Running the Notebooks

The notebooks can be executed in numerical order:

```text
01 → Balance Sheet Analysis
02 → Capital Analysis
03 → DCF Analysis
04 → Profitability Analysis
05 → Company Information
06 → Beneish M-Score
07 → Historical Valuation & Risk/Reward
```

Each notebook imports the shared prepared data through:

```python
from src.setup import *
```

The analytical notebooks operate on stored project data and do not require live market-data requests.

For installation, data sources, repository structure, and overall project methodology, see the [main README](../README.md).
