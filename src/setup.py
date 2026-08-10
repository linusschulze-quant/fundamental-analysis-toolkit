# =========================
# Imports
# =========================

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd


# =========================
# Configuration
# =========================

TICKER: str = "MSFT"

BETA: float = 0.87
RISK_FREE_RATE: float = 0.04
MARKET_RETURN: float = 0.10
TAX_RATE: float = 0.19
CREDIT_SPREAD: float = 0.0005

DATA_START = pd.Timestamp("2005-12-31")
DATA_AS_OF = pd.Timestamp("2025-12-26")


# =========================
# Project Paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "snapshot"

DATA_FILE_1 = RAW_DATA_DIR / "data(1).csv"
DATA_FILE_2 = RAW_DATA_DIR / "data(2).csv"

MARKET_SNAPSHOT_FILE = (
    SNAPSHOT_DIR / "market_snapshot.csv"
)

BALANCE_SHEET_FILE = (
    SNAPSHOT_DIR / "balance_sheet.csv"
)

QUARTERLY_BALANCE_SHEET_FILE = (
    SNAPSHOT_DIR / "quarterly_balance_sheet.csv"
)

INCOME_STATEMENT_FILE = (
    SNAPSHOT_DIR / "income_statement.csv"
)

QUARTERLY_INCOME_STATEMENT_FILE = (
    SNAPSHOT_DIR / "quarterly_income_statement.csv"
)

CASH_FLOW_FILE = (
    SNAPSHOT_DIR / "cash_flow.csv"
)

QUARTERLY_CASH_FLOW_FILE = (
    SNAPSHOT_DIR / "quarterly_cash_flow.csv"
)

COMPANY_PROFILE_FILE = (
    SNAPSHOT_DIR / "company_profile.json"
)

METADATA_FILE = (
    SNAPSHOT_DIR / "metadata.json"
)


# =========================
# Helper Functions
# =========================

def load_financial_csv(
    path: Path,
) -> pd.DataFrame:
    """
    Load and clean a financial CSV file.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}"
        )

    data = pd.read_csv(
        path,
        parse_dates=["Date"],
    )

    if "Date" not in data.columns:
        raise ValueError(
            f"'Date' column missing in {path}"
        )

    data = (
        data
        .drop_duplicates(
            subset="Date",
            keep="last",
        )
        .set_index("Date")
        .sort_index()
    )

    for column in data.columns:

        if not pd.api.types.is_numeric_dtype(
            data[column]
        ):
            data[column] = (
                data[column]
                .astype("string")
                .str.replace(
                    ",",
                    "",
                    regex=False,
                )
                .str.strip()
            )

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    return data


def load_statement_snapshot(
    path: Path,
) -> pd.DataFrame:
    """
    Load a stored financial-statement snapshot.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Snapshot file not found: {path}. "
            "Run scripts/create_snapshot.py first."
        )

    statement = pd.read_csv(
        path,
        index_col=0,
    )

    if statement.empty:
        return statement

    statement.columns = pd.to_datetime(
        statement.columns,
        errors="coerce",
    )

    statement = statement.loc[
        :,
        statement.columns.notna(),
    ]

    return statement.sort_index(
        axis=1,
        ascending=False,
    )


def load_json_file(
    path: Path,
) -> dict:
    """
    Load a JSON file.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Snapshot file not found: {path}. "
            "Run scripts/create_snapshot.py first."
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def last_valid_value(
    data: pd.DataFrame,
    column: str,
) -> float:
    """
    Return the latest non-missing value of a column.
    """

    if column not in data.columns:
        return np.nan

    values = data[column].dropna()

    if values.empty:
        return np.nan

    return float(
        values.iloc[-1]
    )


def trailing_four_quarter_sum(
    data: pd.DataFrame,
    column: str,
) -> float:
    """
    Calculate a trailing-twelve-month value
    from the latest four quarterly observations.
    """

    if column not in data.columns:
        return np.nan

    values = (
        data[column]
        .dropna()
        .tail(4)
    )

    if len(values) < 4:
        return np.nan

    return float(
        values.sum()
    )


# =========================
# Load Raw Financial Data
# =========================

df1 = load_financial_csv(
    DATA_FILE_1
)

df2 = load_financial_csv(
    DATA_FILE_2
)

overlapping_columns = (
    set(df1.columns)
    & set(df2.columns)
)

if overlapping_columns:
    raise ValueError(
        "Duplicate columns found across input files: "
        f"{sorted(overlapping_columns)}"
    )

df_raw = (
    df1
    .join(
        df2,
        how="outer",
    )
    .sort_index()
)

df_raw = df_raw.loc[
    (
        df_raw.index
        >= DATA_START
    )
    & (
        df_raw.index
        <= DATA_AS_OF
    )
].copy()

df_raw["Year"] = (
    df_raw.index.year
)


# =========================
# Column Names
# =========================

ticker = TICKER

quarterly_fcf_col = (
    f"{ticker}: Quarterly Free Cash Flow"
)

quarterly_eps_col = (
    f"{ticker}: Quarterly EPS"
)

cash_col = (
    f"{ticker}: Quarterly Cash & Cash Equivalents"
)

long_term_debt_col = (
    f"{ticker}: Quarterly Long Term Debt"
)

equity_col = (
    f"{ticker}: Quarterly Shareholders Equity"
)

accounts_receivable_col = (
    f"{ticker}: Quarterly Accounts Receivable"
)

shares_col = (
    f"{ticker}: Shares Outstanding"
)

pe_col = (
    f"{ticker}: PE Ratio"
)

ev_ebitda_col = (
    f"{ticker}: EV/EBITDA"
)

market_cap_col = (
    f"{ticker}: Market Cap"
)


# =========================
# Column Validation
# =========================

expected_columns = [
    quarterly_fcf_col,
    quarterly_eps_col,
    cash_col,
    long_term_debt_col,
    equity_col,
    accounts_receivable_col,
    shares_col,
    pe_col,
    ev_ebitda_col,
    market_cap_col,
]

missing_columns = [
    column
    for column in expected_columns
    if column not in df_raw.columns
]

if missing_columns:
    warnings.warn(
        "Missing expected columns:\n"
        + "\n".join(
            missing_columns
        ),
        stacklevel=2,
    )


# =========================
# Working Dataset
# =========================

df = df_raw.copy()

stock_columns = [
    long_term_debt_col,
    equity_col,
    cash_col,
    accounts_receivable_col,
    shares_col,
]

available_stock_columns = [
    column
    for column in stock_columns
    if column in df.columns
]

df[
    available_stock_columns
] = (
    df[
        available_stock_columns
    ]
    .ffill()
)


# =========================
# Annual Flow Data
# =========================

years = sorted(
    df_raw["Year"]
    .dropna()
    .unique()
)

annual_flows = pd.DataFrame(
    index=years
)

annual_flows.index.name = "Year"


if quarterly_fcf_col in df_raw.columns:

    grouped_fcf = (
        df_raw
        .groupby("Year")[
            quarterly_fcf_col
        ]
    )

    annual_fcf = grouped_fcf.sum(
        min_count=1
    )

    annual_fcf = annual_fcf.where(
        grouped_fcf.count() >= 4
    )

    annual_flows[
        f"{ticker}: Annual Free Cash Flow"
    ] = annual_fcf


if quarterly_eps_col in df_raw.columns:

    grouped_eps = (
        df_raw
        .groupby("Year")[
            quarterly_eps_col
        ]
    )

    annual_eps = grouped_eps.sum(
        min_count=1
    )

    annual_eps = annual_eps.where(
        grouped_eps.count() >= 4
    )

    annual_flows[
        f"{ticker}: Annual EPS"
    ] = annual_eps


# =========================
# Year-End Balance Sheet Data
# =========================

balance_columns = [
    long_term_debt_col,
    equity_col,
    cash_col,
    accounts_receivable_col,
    shares_col,
]

available_balance_columns = [
    column
    for column in balance_columns
    if column in df_raw.columns
]

year_end_balance = (
    df_raw
    .groupby("Year")[
        available_balance_columns
    ]
    .last()
)


# =========================
# Annual Valuation Data
# =========================

valuation_columns = [
    pe_col,
    ev_ebitda_col,
    market_cap_col,
]

available_valuation_columns = [
    column
    for column in valuation_columns
    if column in df_raw.columns
]

annual_valuation = (
    df_raw
    .groupby("Year")[
        available_valuation_columns
    ]
    .mean()
)


# =========================
# Combined Annual Data
# =========================

annual_data = pd.concat(
    [
        annual_flows,
        year_end_balance,
        annual_valuation,
    ],
    axis=1,
).sort_index()


# =========================
# Trailing Twelve Months
# =========================

ttm_fcf = (
    trailing_four_quarter_sum(
        df_raw,
        quarterly_fcf_col,
    )
)

ttm_eps = (
    trailing_four_quarter_sum(
        df_raw,
        quarterly_eps_col,
    )
)


# =========================
# Latest Financial Values
# =========================

latest_cash = last_valid_value(
    df_raw,
    cash_col,
)

latest_long_term_debt = (
    last_valid_value(
        df_raw,
        long_term_debt_col,
    )
)

latest_equity = last_valid_value(
    df_raw,
    equity_col,
)

latest_accounts_receivable = (
    last_valid_value(
        df_raw,
        accounts_receivable_col,
    )
)

shares_outstanding = (
    last_valid_value(
        df_raw,
        shares_col,
    )
)

latest_pe = last_valid_value(
    df_raw,
    pe_col,
)

latest_ev_ebitda = (
    last_valid_value(
        df_raw,
        ev_ebitda_col,
    )
)

market_cap = last_valid_value(
    df_raw,
    market_cap_col,
)


# =========================
# Snapshot Metadata
# =========================

snapshot_metadata = load_json_file(
    METADATA_FILE
)

if (
    snapshot_metadata.get("ticker")
    != TICKER
):
    raise ValueError(
        "Snapshot ticker does not match TICKER: "
        f"{snapshot_metadata.get('ticker')} "
        f"!= {TICKER}"
    )

snapshot_date = pd.Timestamp(
    snapshot_metadata[
        "data_as_of"
    ]
)

if snapshot_date != DATA_AS_OF:
    raise ValueError(
        "Snapshot date does not match DATA_AS_OF: "
        f"{snapshot_date.date()} "
        f"!= {DATA_AS_OF.date()}"
    )


# =========================
# Market Snapshot
# =========================

if not MARKET_SNAPSHOT_FILE.exists():
    raise FileNotFoundError(
        f"Snapshot file not found: "
        f"{MARKET_SNAPSHOT_FILE}. "
        "Run scripts/create_snapshot.py first."
    )

market_snapshot = pd.read_csv(
    MARKET_SNAPSHOT_FILE,
    parse_dates=["Date"],
)

market_snapshot = (
    market_snapshot
    .loc[
        market_snapshot["Date"]
        <= DATA_AS_OF
    ]
    .sort_values("Date")
)

if market_snapshot.empty:
    raise ValueError(
        "No market-price observation found "
        "on or before DATA_AS_OF."
    )

price_date = (
    market_snapshot[
        "Date"
    ]
    .iloc[-1]
)

price = float(
    market_snapshot[
        "Close"
    ]
    .iloc[-1]
)


# =========================
# Financial Statement Snapshots
# =========================

balance_sheet = (
    load_statement_snapshot(
        BALANCE_SHEET_FILE
    )
)

quarterly_balance_sheet = (
    load_statement_snapshot(
        QUARTERLY_BALANCE_SHEET_FILE
    )
)

income_statement = (
    load_statement_snapshot(
        INCOME_STATEMENT_FILE
    )
)

quarterly_income_statement = (
    load_statement_snapshot(
        QUARTERLY_INCOME_STATEMENT_FILE
    )
)

cash_flow = (
    load_statement_snapshot(
        CASH_FLOW_FILE
    )
)

quarterly_cash_flow = (
    load_statement_snapshot(
        QUARTERLY_CASH_FLOW_FILE
    )
)

company_profile = load_json_file(
    COMPANY_PROFILE_FILE
)


# =========================
# Risk Parameters
# =========================

market_risk_premium = (
    MARKET_RETURN
    - RISK_FREE_RATE
)

equity_risk_premium = (
    BETA
    * market_risk_premium
)