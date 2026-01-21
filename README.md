# fundamental-analysis-toolkit

A modular, professional-grade Python framework for conducting systematic fundamental equity analysis.

This repository provides reusable financial analysis functions and well-documented notebooks covering financial statements, valuation, risk/reward analysis, and forensic accounting techniques.

🚀 Features

Company Information & Financial Overview

Balance Sheet Analysis

Capital Structure & Leverage Analysis

Discounted Cash Flow (DCF) Valuation

Profitability & Efficiency Metrics

Beneish M-Score (Earnings Manipulation Detection)

Risk/Reward & Valuation Range Analysis (EV/EBITDA & P/E)

Probability-Adjusted Risk/Reward Ratios

Backtesting Framework (in progress)

🧱 Project Structure
fundamental-analysis-toolkit/
│
├── src/                # Core analysis logic (reusable functions)
├── notebooks/         # Step-by-step analysis workflows
├── data/              # Raw and processed financial data
├── figures/           # Optional output outputs
├── requirements.txt
└── README.md

⚙️ Installation
git clone https://github.com/linuschulze-quant/fundamental-analysis-toolkit.git
cd fundamental-analysis-toolkit
pip install -r requirements.txt

▶️ Usage

Set the ticker and parameters in src/setup.py.

Load financial data from the data/ directory.

Run the notebooks in the notebooks/ folder in numerical order.

Review outputs directly in the notebooks.


📊 Data Sources

Financial statement data is sourced from:

Wall Street Numbers (CSV exports)

Yahoo Finance via the yfinance API

All datasets are used for research and educational purposes only.

🎯 Project Goals

Build a transparent, reproducible valuation framework.

Separate financial logic from presentation and experimentation.

Enable scalable equity research workflows.

Support both discretionary and systematic investment analysis.

👥 Target Audience

Quantitative analysts

Fundamental investors

Finance students

Researchers building valuation pipelines

⚠️ Disclaimer

This project is for educational and research purposes only and does not constitute financial advice.

📬 Contributions

Pull requests and feedback are welcome. This project is under active development.
