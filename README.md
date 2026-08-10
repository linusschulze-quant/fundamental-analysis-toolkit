# Fundamental Analysis Toolkit

A reproducible Python framework for fundamental equity analysis, combining financial statement analysis, valuation, profitability, forensic accounting, and historical valuation research.

The current implementation applies the framework to **Microsoft Corporation (MSFT)** using fixed historical data and stored financial-statement snapshots. The analytical notebooks do not require live market-data requests, ensuring that results remain transparent and reproducible.

## Overview

The toolkit contains seven analysis modules:

| Notebook                          | Focus                                                               |
| --------------------------------- | ------------------------------------------------------------------- |
| `01_balance_sheet_analysis.ipynb` | Balance sheet structure, leverage, and net debt                     |
| `02_capital_analysis.ipynb`       | Capital structure, WACC, and financing efficiency                   |
| `03_dcf_analysis.ipynb`           | Discounted Cash Flow valuation                                      |
| `04_profitability_analysis.ipynb` | Margins, returns, cash generation, and growth                       |
| `05_company_info.ipynb`           | Company, market, and book-value information                         |
| `06_beneish_m_score.ipynb`        | Beneish M-Score forensic accounting analysis                        |
| `07_risk_reward_analysis.ipynb`   | Historical P/E and EV/EBITDA distributions and risk/reward analysis |

For a detailed explanation of each module and its methodology, see [`notebooks/README.md`](notebooks/README.md).

## Methodology

The project separates **data acquisition from analysis**.

Instead of downloading new data whenever a notebook is executed, the analysis uses stored datasets with defined reporting dates. This prevents historical outputs from changing because of later data revisions and makes the calculations reproducible.

Particular attention is paid to distinguishing:

* annual financial-statement data
* quarterly balance-sheet data
* trailing-twelve-month metrics
* historical valuation series
* point-in-time market data

The current dataset includes market data through **December 26, 2025** and the latest quarterly financial-statement snapshot through **September 30, 2025**.

## Project Structure

```text
fundamental-analysis-toolkit/
├── data/
│   ├── raw/            # Historical financial and valuation series
│   └── snapshot/       # Fixed financial-statement and company snapshots
│
├── notebooks/
│   ├── 01_balance_sheet_analysis.ipynb
│   ├── 02_capital_analysis.ipynb
│   ├── 03_dcf_analysis.ipynb
│   ├── 04_profitability_analysis.ipynb
│   ├── 05_company_info.ipynb
│   ├── 06_beneish_m_score.ipynb
│   ├── 07_risk_reward_analysis.ipynb
│   └── README.md
│
├── src/
│   └── setup.py        # Shared configuration and data preparation
│
├── requirements.txt
└── README.md
```

## Key Features

* Financial statement and capital structure analysis
* CAPM-based cost of equity and WACC calculation
* Discounted Cash Flow valuation
* Profitability, efficiency, and cash-flow metrics
* Beneish M-Score earnings-manipulation screening
* Historical P/E and EV/EBITDA distribution analysis
* Valuation-based WC / Buy / Fair Value / Sell reference levels
* Empirical historical valuation frequencies
* Explicit handling of different reporting periods
* Fixed-input architecture for reproducible analysis

## Installation

```bash
git clone https://github.com/linusschulze-quant/fundamental-analysis-toolkit.git
cd fundamental-analysis-toolkit
pip install -r requirements.txt
```

Required analytical dependencies:

```text
numpy
pandas
```

Open the repository in a Jupyter-compatible environment such as JupyterLab, Jupyter Notebook, or VS Code.

## Usage

1. Review the shared configuration and data preparation in `src/setup.py`.
2. Open the notebooks in `notebooks/`.
3. Run the notebooks in numerical order or execute individual analyses independently.
4. Review methodology, outputs, and limitations directly inside each notebook.

The current repository is configured for **MSFT**, but the analytical structure is designed to provide a foundation for applying the same approach to other companies.

## Data Sources

Stored project data originates from:

* [Wall Street Numbers](https://wallstreetnumbers.com/) — historical financial and valuation series
* [Yahoo Finance](https://finance.yahoo.com/) — financial-statement and company snapshots originally retrieved via `yfinance`

The analytical notebooks operate on the stored files and therefore do **not** require live Yahoo Finance requests.

Snapshot metadata is documented in:

```text
data/snapshot/metadata.json
```

## Project Goals

I built this project to develop a more systematic approach to equity research while combining financial analysis with Python.

The main objectives are to:

* make valuation calculations transparent
* distinguish financial data across reporting periods correctly
* build reproducible analytical workflows
* combine fundamental and historical valuation analysis
* create a foundation for further work in quantitative finance and systematic equity research

## Limitations

This repository is an analytical and educational framework rather than a complete institutional valuation model.

Model outputs depend on their underlying assumptions and data definitions. Historical valuation levels are descriptive rather than predictive, DCF results are sensitive to growth and discount-rate assumptions, and financial-statement and market-data dates do not always coincide.

## Disclaimer

This project is for educational and research purposes only and does **not constitute financial advice**.

All calculations and model outputs should be independently evaluated before being used for any financial decision.
