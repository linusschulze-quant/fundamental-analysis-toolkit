

import pandas as pd
import numpy as np

# =========================
# Balance Sheet Analysis
# =========================

def balance_sheet_analysis(df, balance_sheet, ticker):
    """
    Compute key balance sheet metrics using latest quarterly data.

    Parameters
    ----------
    df : pandas.DataFrame
        Quarterly financial data with date index.
    balance_sheet : pandas.DataFrame
        Latest balance sheet data from yfinance.
    ticker : str
        Stock ticker symbol.

    Returns
    -------
    dict
        Dictionary with total assets, debt, equity, cash, net debt, and ratios.
    """
    try:
        equity = df[f"{ticker}: Quarterly Shareholders Equity"].iloc[-1] # Latest equity value
        debt = balance_sheet.loc["Total Debt"].iloc[0] # Latest total debt from balance sheet
        cash = balance_sheet.loc["Cash And Cash Equivalents"].iloc[0] # Latest cash from balance sheet
        long_term_debt = df[f"{ticker}: Quarterly Long Term Debt"].iloc[-1] # Latest long term debt value
        interest_bearing_debt = long_term_debt # Assuming long term debt is interest bearing
    except KeyError as e:
        raise KeyError(f"Missing required column in data: {e}")

    net_debt = interest_bearing_debt - cash
    gearing_ratio = net_debt / equity if equity != 0 else np.nan
    equity_ratio = equity / (debt + equity) if (debt + equity) != 0 else np.nan
    debt_ratio = debt / (debt + equity) if (debt + equity) != 0 else np.nan
    total_assets = debt + equity

    return {
        "Total Assets": total_assets,
        "Debt": debt,
        "Equity": equity,
        "Cash": cash,
        "Long Term Debt": long_term_debt,
        "Net Debt": net_debt,
        "Gearing Ratio": gearing_ratio,
        "Equity Ratio": equity_ratio,
        "Debt Ratio": debt_ratio
    }




# =========================
# Capital Analysis
# =========================


def capital_analysis(df, balance_sheet, financials, market_cap, risk_free_rate, beta,
                     market_risk_premium, credit_spread, tax_rate, ticker, interest_expense=None):
    """
    Compute capital structure and WACC metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        Quarterly balance sheet data.
    balance_sheet : pandas.DataFrame
        Latest balance sheet data.
    financials : pandas.DataFrame
        Yahoo Finance financials data.
    market_cap : float
        Current market capitalization of the company.
    risk_free_rate : float
        Risk-free interest rate (as decimal or percent depending on usage).
    beta : float
        Stock beta for CAPM calculation.
    market_risk_premium : float
        Expected market risk premium.
    credit_spread : float
        Credit spread for debt cost calculation.
    tax_rate : float
        Corporate tax rate in percent.
    ticker : str
        Stock ticker symbol.
    interest_expense : float, optional
        Interest expense value; if None, taken from financials.

    Returns
    -------
    tuple
        df_results : pandas.DataFrame
            Formatted table of calculated metrics.
        results : dict
            Dictionary with numeric metric values.
    """
    
    # --- Get latest financial values safely ---
    equity = df[f"{ticker}: Quarterly Shareholders Equity"].iloc[-1]
    debt = balance_sheet.loc["Total Debt"].iloc[0]
    revenue = financials.loc["Total Revenue"].iloc[0]
    ebit = financials.loc["EBIT"].iloc[0]
    
    # Use provided interest expense or fallback to financials
    if interest_expense is None:
        interest_expense = financials.loc["Interest Expense"].iloc[0] 
    
     # Ratios for WACC calculation
    total_assets = debt + equity
    equity_ratio = equity / total_assets if total_assets != 0 else np.nan
    debt_ratio = debt / total_assets if total_assets != 0 else np.nan
    
    equity_ratio_wacc = market_cap / (market_cap + debt) if (market_cap + debt) != 0 else np.nan
    debt_ratio_wacc = debt / (market_cap + debt) if (market_cap + debt) != 0 else np.nan
    
    # --- Cost calculations ---
    cost_of_debt = (risk_free_rate + credit_spread) * (1 - tax_rate / 100)
    cost_of_equity = risk_free_rate + beta * market_risk_premium
    wacc = equity_ratio_wacc * cost_of_equity + debt_ratio_wacc * cost_of_debt
    
    # --- Other metrics ---
    revenue_to_capital = revenue / total_assets if total_assets != 0 else np.nan
    interest_coverage = ebit / interest_expense if interest_expense != 0 else np.nan
    
    results = {
        "Total Assets": total_assets,
        "Debt": debt,
        "Equity": equity,
        "Equity Ratio": equity_ratio,
        "Debt Ratio": debt_ratio,
        "Cost of Equity": cost_of_equity,
        "Cost of Debt": cost_of_debt,
        "WACC": wacc,
        "Revenue to Capital": revenue_to_capital,
        "Interest Coverage Ratio": interest_coverage
    }
    
    # --- Convert to DataFrame for nice display ---
    df_results = pd.DataFrame(results, index=[0]).T.rename(columns={0: "Value"})
    df_results["Value"] = df_results["Value"].apply(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)
    
    return df_results, results




# =========================
# Discounted Cash Flow (DCF) Analysis
# =========================


def calculate_dcf(fcf, cash, debt, discount_rate, growth, terminal_growth, years=10):
    """
    Compute the DCF Fair Value of a company.

    Parameters
    ----------
    fcf : float
        Latest Free Cash Flow.
    cash : float
        Latest cash & cash equivalents.
    debt : float
        Latest total debt.
    discount_rate : float
        Discount rate (WACC).
    growth : float
        Expected annual FCF growth (decimal, e.g., 0.05 for 5%).
    terminal_growth : float
        Long-term growth rate (decimal).
    years : int, optional
        Projection period (default 10).
    
    Returns
    -------
    fair_value : float
        Total DCF Fair Value
    discounted_cashflows : list
        Present value of FCFs per year
    discounted_terminal_value : float
        Present value of Terminal Value
    """

    
    if terminal_growth >= discount_rate:
        raise ValueError("Terminal growth must be smaller than discount rate.")
    
    discounted_cashflows = []
    fcf_year = fcf
    
    for year in range(1, years + 1):
        fcf_year *= (1 + growth) # project FCF growth
        discounted_cashflows.append(fcf_year / (1 + discount_rate)**year)
    
    terminal_value = fcf_year / (discount_rate - terminal_growth) # Terminal Value at end of projection
    discounted_terminal_value = terminal_value / (1 + discount_rate)**years
    
    # Total fair value includes cash, subtracts debt
    fair_value = sum(discounted_cashflows) + discounted_terminal_value + cash - debt
    
    return fair_value, discounted_cashflows, discounted_terminal_value



# =========================
# Historical DCF Calculation (Optional)
# =========================

def historical_dcf(annual_avg, growth, terminal_growth, discount_rate):
    """
    Compute DCF Fair Value historically for each year.
    Returns a DataFrame with Fair Values per historical year.

    Parameters
    ----------
    annual_avg : pandas.DataFrame
        Annualized financial data including Free Cash Flow, Cash, and Debt.
    growth : float
        Expected annual Free Cash Flow growth (decimal, e.g., 0.05 for 5%).
    terminal_growth : float
        Long-term growth rate (decimal, e.g., 0.03 for 3%).
    discount_rate : float
        Discount rate to apply (typically WACC).

    Returns
    -------
    pandas.DataFrame
        DataFrame with historical years as index and the calculated
        DCF Fair Value for each year in a column named
        "Historical Fair Value ($)".

    Note:
        Historical fair values are **discounted using today's discount rate and growth assumptions**.
        They **do not represent the actual fair value at that historical date** and should only
        be used for visualization or illustrative purposes.

    """

    historical_cash = annual_avg[f"{TICKER}: Quarterly Cash & Cash Equivalents"].tolist()
    historical_debt = annual_avg[f"{TICKER}: Quarterly Long Term Debt"].tolist()
    historical_fcf = annual_avg[f"{TICKER}: Annual Free Cash Flow"].tolist()
    years = annual_avg.index.tolist()
    
    fair_values = []
    for i in range(len(years)):
        fv, _, _ = calculate_dcf(
            fcf=historical_fcf[i],
            cash=historical_cash[i],
            debt=historical_debt[i],
            growth=growth,
            terminal_growth=terminal_growth,
            discount_rate=discount_rate,
            years=10
        )
        fair_values.append(fv)
    
    df_hist = pd.DataFrame({
        "Year": years,
        "Historical Fair Value ($)": fair_values
    })
    df_hist.set_index("Year", inplace=True)
    df_hist["Historical Fair Value ($)"] = df_hist["Historical Fair Value ($)"].apply(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)
    return df_hist




# =========================
# Profitability Analysis
# =========================

def profitability_analysis(df, annual_avg, info, balance_sheet, ticker, market_cap, price, shares_outstanding, tax_rate, eps_growth_forward=None):
    """
    Compute profitability, margin, return and valuation metrics.

    Parameters:
    ----------
    df : pandas.DataFrame
        Quarterly balance sheet data.
    annual_avg : pandas.DataFrame
        Annual averages of key metrics, e.g., FCF, EPS.
    info : yfinance.Ticker
        Financial information from Yahoo Finance.
    balance_sheet : pandas.DataFrame
        Latest balance sheet data.
    ticker : str
        Stock ticker symbol.
    market_cap : float
        Current market capitalization.
    price : float
        Current stock price.
    shares_outstanding : float
        Latest number of shares outstanding.
    tax_rate : float
        Corporate tax rate in percent.
    eps_growth_forward : float, optional
        Forward EPS growth for PEG calculation (in percent).

    Returns
    -------
    pd.DataFrame
        Key profitability and valuation metrics.
    """

    # --- Core financials ---
    revenue = info.financials.loc["Total Revenue"].iloc[0] # Latest revenue
    gross_profit = info.financials.loc["Gross Profit"].iloc[0] # Latest gross profit
    net_income = info.financials.loc["Net Income"].iloc[0] # Latest net income  
    ebit = info.financials.loc["EBIT"].iloc[0] # Latest EBIT
    ebitda = info.financials.loc["EBITDA"].iloc[0] # Latest EBITDA
    free_cashflow = annual_avg[f"{ticker}: Annual Free Cash Flow"].iloc[-1] # Latest FCF

    equity = df[f"{ticker}: Quarterly Shareholders Equity"].iloc[-1] # Latest equity value
    debt = balance_sheet.loc["Total Debt"].iloc[0] # Latest debt
    cash = annual_avg[f"{ticker}: Quarterly Cash & Cash Equivalents"].iloc[-1] # Latest cash

    # --- Valuation ---
    eps = net_income / shares_outstanding if shares_outstanding != 0 else np.nan #calculate EPS
    pe_ratio = price / eps if eps != 0 else np.nan 
    earnings_yield = eps / price if price != 0 else np.nan

    # --- Margins ---
    gross_margin = gross_profit / revenue if revenue != 0 else np.nan
    net_margin = net_income / revenue if revenue != 0 else np.nan
    ebit_margin = ebit / revenue if revenue != 0 else np.nan
    ebitda_margin = ebitda / revenue if revenue != 0 else np.nan

    # --- Returns ---
    total_assets = equity + debt
    nopat = ebit * (1 - tax_rate / 100)
    invested_capital = total_assets - cash

    roe = net_income / equity if equity != 0 else np.nan # Return on Equity
    roa = net_income / total_assets if total_assets != 0 else np.nan # Return on Assets
    roic = nopat / invested_capital if invested_capital != 0 else np.nan # Return on Invested Capital

    # --- Cashflow metrics ---
    fcf_yield = free_cashflow / market_cap if market_cap != 0 else np.nan
    operating_leverage = ebit / gross_profit if gross_profit != 0 else np.nan

    # --- EPS growth (trailing) ---
    eps_series = annual_avg[f"{ticker}: Annual EPS"].dropna()

    trailing_growth = None
    trailing_peg = None
    if len(eps_series) >= 4:
        growth_rates = [
            eps_series.iloc[-1] / eps_series.iloc[-2] - 1,
            eps_series.iloc[-2] / eps_series.iloc[-3] - 1,
            eps_series.iloc[-3] / eps_series.iloc[-4] - 1,
        ]
        trailing_growth = np.mean(growth_rates)
        trailing_peg = pe_ratio / (trailing_growth * 100) # multiply by 100 to convert to percent

    # --- Forward PEG ---
    forward_peg = None
    if eps_growth_forward is not None:
        forward_peg = pe_ratio / eps_growth_forward

    # --- Free Cashflow Growth ---
    fcf_growth = None
    fcf_series = annual_avg.get(f"{ticker}: Annual Free Cash Flow", pd.Series([np.nan]))
    if len(fcf_series) >= 2:
        prev_fcf = fcf_series.iloc[-2]
        fcf_growth = (free_cashflow / prev_fcf - 1) if prev_fcf != 0 else np.nan


    # --- Output ---
    metrics = {
        "EPS": eps,
        "P/E Ratio": pe_ratio,
        "Earnings Yield": earnings_yield,
        "Gross Margin": gross_margin,
        "Net Margin": net_margin,
        "EBIT Margin": ebit_margin,
        "EBITDA Margin": ebitda_margin,
        "Free Cash Flow Yield": fcf_yield,
        "Operating Leverage": operating_leverage,
        "Return on Equity (ROE)": roe,
        "Return on Assets (ROA)": roa,
        "ROIC": roic,
        "Trailing PEG Ratio": trailing_peg,
        "Forward PEG Ratio": forward_peg,
        "Average EPS Growth (3 Years)": trailing_growth,
        "Free Cashflow Growth (1 Year)": fcf_growth,
        "Total Assets": total_assets,
        "Revenue": revenue,
        "Net Income": net_income,
        "EBIT": ebit,
        "EBITDA": ebitda
    }
     
    df_metrics = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
    return df_metrics



# =========================
# Company Information
# =========================

def company_information(info, market_cap):
    """
    Collect general company and market information from Yahoo Finance.

    Parameters
    ----------
    info : yfinance.Ticker
        Ticker object with company information.
    market_cap : float
        Current market capitalization of the company.

    Returns
    -------
    pd.DataFrame
        Company metadata and key descriptive metrics.
    """
    
    # Helper to safely get values from Yahoo Finance info dictionary
    def safe_get(key):
        return info.info.get(key, None)

    # Collect data into dictionary
    data = {
        "Company Name": safe_get("longName"),
        "Country": safe_get("country"),
        "Sector": safe_get("sector"),
        "Industry": safe_get("industry"),
        "Currency": safe_get("currency"),

        "Market Capitalization": market_cap,
        "Average Volume": safe_get("averageVolume"),

        "All-Time High": safe_get("allTimeHigh"),
        "All-Time Low": safe_get("allTimeLow"),

        "Analyst Rating": safe_get("recommendationKey"),
        "Target Price (Median)": safe_get("targetMedianPrice"),
        "Number of Analysts": safe_get("numberOfAnalystOpinions"),

        "Insider Ownership (%)": (
            safe_get("heldPercentInsiders") * 100
            if safe_get("heldPercentInsiders") is not None else None
        ),
        "Institutional Ownership (%)": (
            safe_get("heldPercentInstitutions") * 100
            if safe_get("heldPercentInstitutions") is not None else None
        ),

        "Shares Short": safe_get("sharesShort"),
        "Book Value per Share": safe_get("bookValue"),
        "Price-to-Book Ratio": safe_get("priceToBook"),
    }

     # Convert dictionary to DataFrame for display
    df_info = pd.DataFrame.from_dict(data, orient="index", columns=["Value"])
    return df_info




# =========================
# Beneish M-Score Calculation 
# =========================

def calculate_beneish_m_score(df, info, balance_sheet, ticker):
    """
    Calculate the Beneish M-Score to detect potential earnings manipulation.

    Parameters
    ----------
    df : pandas.DataFrame
        Quarterly financial dataframe with datetime index.
    info : object
        Financial data provider (e.g. yfinance.Ticker object).
    balance_sheet : pandas.DataFrame
        Balance sheet data (latest year first).
    ticker : str
        Stock ticker symbol.

    Returns
    -------
    float
        Beneish M-Score

    Info
    ----
    M-Score < -2.22 → low likelihood of manipulation
    M-Score > -2.22 → possible earnings manipulation
    """
    # Ensure we have at least two years of data
    years = sorted(df.index.year.unique())
    if len(years) < 2:
        raise ValueError("At least two years of data are required")

    t, t1 = years[-1], years[-2]

    # --- DSRI: Days Sales in Receivables Index ---
    ar_t = df.loc[str(t), f"{ticker}: Quarterly Accounts Receivable"].iloc[-1]
    ar_t1 = df.loc[str(t1), f"{ticker}: Quarterly Accounts Receivable"].iloc[-1]

    revenue_t = info.financials.loc["Total Revenue"].iloc[0]
    revenue_t1 = info.financials.loc["Total Revenue"].iloc[1]

    dsri = (ar_t / revenue_t) / (ar_t1 / revenue_t1) if revenue_t1 != 0 and revenue_t != 0 else np.nan

    # --- GMI: Gross Margin Index ---
    gp_t = info.financials.loc["Gross Profit"].iloc[0]
    gp_t1 = info.financials.loc["Gross Profit"].iloc[1]

    gross_margin_t = gp_t / revenue_t if revenue_t != 0 else np.nan
    gross_margin_t1 = gp_t1 / revenue_t1  if revenue_t1 != 0 else np.nan
    gmi = gross_margin_t1 / gross_margin_t if gross_margin_t != 0 else np.nan

    # --- SGI: Sales Growth Index ---
    sgi = revenue_t / revenue_t1

    # --- AQI: Asset Quality Index ---
    current_assets_t = balance_sheet.loc["Current Assets"].iloc[0]
    current_assets_t1 = balance_sheet.loc["Current Assets"].iloc[1]

    ppe_t = balance_sheet.loc["Net PPE"].iloc[0]
    ppe_t1 = balance_sheet.loc["Net PPE"].iloc[1]

    total_assets_t = balance_sheet.loc["Total Assets"].iloc[0]
    total_assets_t1 = balance_sheet.loc["Total Assets"].iloc[1]

    aqi = (
        1 - ((current_assets_t + ppe_t) / total_assets_t)
    ) / (
        1 - ((current_assets_t1 + ppe_t1) / total_assets_t1)
    ) if total_assets_t != 0 and total_assets_t1 != 0 else np.nan

    # --- DEPI: Depreciation Index ---
    depreciation_t = abs(balance_sheet.loc["Accumulated Depreciation"].iloc[0])
    depreciation_t1 = abs(balance_sheet.loc["Accumulated Depreciation"].iloc[1])

    depi = (
        depreciation_t1 / (depreciation_t1 + ppe_t1)
    ) / (
        depreciation_t / (depreciation_t + ppe_t)
    ) if (depreciation_t + ppe_t) != 0 and (depreciation_t1 + ppe_t1) != 0 else np.nan


    # --- SGAI: Sales, General and Administration Index ---
    sga_t = info.financials.loc["Selling General And Administration"].iloc[0]
    sga_t1 = info.financials.loc["Selling General And Administration"].iloc[1]

    sgai = (sga_t / revenue_t) / (sga_t1 / revenue_t1) if revenue_t != 0 and revenue_t1 != 0 else np.nan

    # --- TATA: Total Accruals to Total Assets ---
    net_income_t = info.financials.loc["Net Income"].iloc[0]
    cfo_t = info.cashflow.loc["Operating Cash Flow"].iloc[0]

    tata = (net_income_t - cfo_t) / total_assets_t if total_assets_t != 0 else np.nan

    # --- LVGI: Leverage Index ---
    debt_t = balance_sheet.loc["Total Debt"].iloc[0]
    debt_t1 = balance_sheet.loc["Total Debt"].iloc[1]

    lvgi = (debt_t / total_assets_t) / (debt_t1 / total_assets_t1) if total_assets_t != 0 and total_assets_t1 != 0 else np.nan

    # --- Beneish M-Score calculation ---
    m_score = (
        -4.84
        + 0.92 * dsri
        + 0.528 * gmi
        + 0.404 * aqi
        + 0.892 * sgi
        + 0.115 * depi
        - 0.172 * sgai
        + 4.679 * tata
        - 0.327 * lvgi
    )

    return m_score



# =========================
# Risk-Reward Ratio Calculation based on EV/EBITDA and PE Ratio
# =========================

def calculate_crv(price, ev_ebitda_list, pe_ratio_list, ebitda, net_debt, eps, shares_outstanding, has_ev=False, has_pe=False):
    """
    Calculate the Risk-Reward Ratio (CRV) based on EV/EBITDA and PE Ratio.

    Parameters
    ----------
    price : float
        Current stock price.
    ev_ebitda_list : list of float
        Historical yearly EV/EBITDA values, sorted from smallest to largest.
    pe_ratio_list : list of float
        Historical yearly PE Ratio values, sorted from smallest to largest.
    ebitda : float
        Current EBITDA of the company.
    net_debt : float
        Net debt of the company (total debt minus cash).
    eps : float
        Earnings per share.
    shares_outstanding : float
        Number of shares outstanding.
    has_ev : bool, default False
        Whether EV/EBITDA data is available.
    has_pe : bool, default False
        Whether PE Ratio data is available.

    Returns
    -------
    dict
        Dictionary containing:
        - pe_values, pe_prices : DataFrames with PE levels and prices
        - ev_values, ev_prices : DataFrames with EV/EBITDA levels and prices
        - df_result, df_prices : combined DataFrames
        - Crv_pe, Crv_ev : calculated Risk-Reward Ratios
        - Chance_ev, Risk_ev, Chance_pe, Risk_pe : upside/downside values
    """

    if not has_pe and not has_ev:
        raise ValueError("Neither EV/EBITDA nor PE Ratio data is available.")

    # Initialize outputs
    pe_values = pe_prices = ev_values = ev_prices = None
    Crv_pe = Crv_ev = None
    Chance_pe = Risk_pe = Chance_ev = Risk_ev = None

    # ---- EV/EBITDA ----
    if has_ev:

        # EV/EBITDA reference values based on sorted historical data
        WC_Ev = round((ev_ebitda_list[0] + np.mean(ev_ebitda_list[1:4])) / 2)  # "Worst Case": minimum + lowest 2-4 values average
        Buy_EV = np.mean(ev_ebitda_list[1:4])  # Buy level: average of lowest 2-4 historical values
        Sell_EV = np.mean(ev_ebitda_list[-4:-1])  # Sell level: average of 2-4 highest historical values
        FV_Ev = (Buy_EV + Sell_EV) / 2  # Fair Value: midpoint between Buy and Sell levels


        WC_EV_Price = (ebitda * WC_Ev - net_debt) / shares_outstanding
        Buy_EV_Price = (ebitda * Buy_EV - net_debt) / shares_outstanding
        FV_EV_Price = (ebitda * FV_Ev - net_debt) / shares_outstanding
        Sell_EV_Price = (ebitda * Sell_EV - net_debt) / shares_outstanding

        ev_values = pd.DataFrame({
            "EV/EBITDA": [WC_Ev, Buy_EV, FV_Ev, Sell_EV]
        }, index=["WC", "Buy", "FV", "Sell"])

        ev_prices = pd.DataFrame({
            "EV/EBITDA": [WC_EV_Price, Buy_EV_Price, FV_EV_Price, Sell_EV_Price]
        }, index=["WC", "Buy", "FV", "Sell"])

        if WC_EV_Price < price and Sell_EV_Price < price:
            Chance_ev = 0
            Risk_ev = price - WC_EV_Price
            Crv_ev = 0
        elif WC_EV_Price > price and Sell_EV_Price > price:
            Chance_ev = Sell_EV_Price - price
            Risk_ev = 0
            Crv_ev = float('inf')
        else:
            Chance_ev = Sell_EV_Price - price
            Risk_ev = price - WC_EV_Price
            Crv_ev = Chance_ev / Risk_ev if Risk_ev != 0 else float('inf')

    # ---- PE Ratio ----
    if has_pe:
        WC_Pe = round((pe_ratio_list[0] + np.mean(pe_ratio_list[1:4])) / 2)
        Buy_Pe = np.mean(pe_ratio_list[1:4])
        Sell_Pe = np.mean(pe_ratio_list[-4:-1])
        FV_Pe = (Buy_Pe + Sell_Pe) / 2

        WC_Pe_Price = eps * WC_Pe
        Buy_Pe_Price = eps * Buy_Pe
        FV_Pe_Price = eps * FV_Pe
        Sell_Pe_Price = eps * Sell_Pe

        pe_values = pd.DataFrame({
            "PE Ratio": [WC_Pe, Buy_Pe, FV_Pe, Sell_Pe]
        }, index=["WC", "Buy", "FV", "Sell"])

        pe_prices = pd.DataFrame({
            "PE Ratio": [WC_Pe_Price, Buy_Pe_Price, FV_Pe_Price, Sell_Pe_Price]
        }, index=["WC", "Buy", "FV", "Sell"])

        if WC_Pe_Price < price and Sell_Pe_Price < price:
            Chance_pe = 0
            Risk_pe = price - WC_Pe_Price
            Crv_pe = 0
        elif WC_Pe_Price > price and Sell_Pe_Price > price:
            Chance_pe = Sell_Pe_Price - price
            Risk_pe = 0
            Crv_pe = float('inf')
        else:
            Chance_pe = Sell_Pe_Price - price
            Risk_pe = price - WC_Pe_Price
            Crv_pe = Chance_pe / Risk_pe if Risk_pe != 0 else float('inf')

    # ---- Combine Results ----
    df_result = pd.concat([pe_values, ev_values], axis=1)
    df_prices = pd.concat([pe_prices, ev_prices], axis=1)

    return {
        "pe_values": pe_values,
        "pe_prices": pe_prices,
        "ev_values": ev_values,
        "ev_prices": ev_prices,
        "df_result": df_result,
        "df_prices": df_prices,
        "Crv_pe": Crv_pe,
        "Crv_ev": Crv_ev,
        "Chance_ev": Chance_ev,
        "Risk_ev": Risk_ev,
        "Chance_pe": Chance_pe,
        "Risk_pe": Risk_pe
    }



# =========================
# Histogram Functions
# =========================

def ev_histogram(ev_series, bins=15):
    """
    Create a histogram table for EV/EBITDA values.

    Parameters
    ----------
    ev_series : pd.Series 
        Historical EV/EBITDA values
    bins : int 
        Number of bins to divide the data into (default: 15)

    Returns
    -------
    hist_ev : pd.DataFrame 
        Histogram table with columns:
        - "Low": lower edge of the bin
        - "High": upper edge of the bin
        - "Number of Days": count of observations in the bin
        - "% of Days": percentage of total observations in the bin
        - "Cumulative (%)": cumulative percentage up to this bin
        - "Probability Up (%)": probability of EV/EBITDA being higher than this bin
    """
    counts, edges = np.histogram(ev_series, bins=bins)
    total_days = len(ev_series)

    hist_ev = pd.DataFrame({
        "Low": edges[:-1],
        "High": edges[1:],
        "Number of Days": counts
    })
    hist_ev["% of Days"] = hist_ev["Number of Days"] / total_days * 100
    hist_ev["Cumulative (%)"] = hist_ev["% of Days"].cumsum().round(2)
    hist_ev["Probability Up (%)"] = (100 - hist_ev["Cumulative (%)"]).round(2)
    return hist_ev


def pe_histogram(pe_series, bins=15):
    """
    Create a histogram table for PE Ratio values.

    Parameters
    ----------
    pe_series : pd.Series
        Historical PE Ratio values
    bins : int 
        Number of bins to divide the data into (default: 15)

    Returns
    -------
    hist_pe : pd.DataFrame
        Histogram table with columns:
        - "Low": lower edge of the bin
        - "High": upper edge of the bin
        - "Number of Days": count of observations in the bin
        - "% of Days": percentage of total observations in the bin
        - "Cumulative (%)": cumulative percentage up to this bin
        - "Probability Up (%)": probability of PE Ratio being higher than this bin
    """

    counts, edges = np.histogram(pe_series, bins=bins)
    total_days = len(pe_series)

    hist_pe = pd.DataFrame({
        "Low": edges[:-1],
        "High": edges[1:],
        "Number of Days": counts
    })
    hist_pe["% of Days"] = hist_pe["Number of Days"] / total_days * 100
    hist_pe["Cumulative (%)"] = hist_pe["% of Days"].cumsum().round(2)
    hist_pe["Probability Up (%)"] = (100 - hist_pe["Cumulative (%)"]).round(2)
    return hist_pe




# =========================
# Probability-Adjusted Risk-Reward Ratio
# =========================

def probability_adjusted_ev(ev_series, current_ev_ebitda, price, ev_ebitda_list, ebitda, net_debt, eps, shares_outstanding):
   
    """
    Calculate probability-adjusted Risk/Reward Ratio (CRV) based on EV/EBITDA.

    Parameters
    ----------
    ev_series : pd.Series
        Historical EV/EBITDA values.
    current_ev_ebitda : float
        Current EV/EBITDA of the company.
    price : float
        Current stock price.
    ev_ebitda_list : list of float
        Historical yearly EV/EBITDA values, sorted.
    ebitda : float
        Current EBITDA value.
    net_debt : float
        Net debt (total debt minus cash).
    eps : float
        Earnings per share.
    shares_outstanding : int
        Number of shares outstanding.

    Returns
    -------
    dict 
        Contains probability-adjusted CRV calculations:
        - "P_up": probability of upside
        - "P_down": probability of downside
        - "Expected_Value": expected monetary value
        - "Upside_adjusted": upside weighted by probability
        - "Downside_adjusted": downside weighted by probability
        - "CRV_adjusted": probability-adjusted Risk/Reward Ratio
    """

    hist_ev = ev_histogram(ev_series, bins=15)

    P_down = hist_ev.loc[hist_ev["High"] <= current_ev_ebitda, "% of Days"].sum() / 100
    P_up = hist_ev.loc[hist_ev["Low"] > current_ev_ebitda, "% of Days"].sum() / 100

    crv_data = calculate_crv(price, ev_ebitda_list, None, ebitda, net_debt, eps, shares_outstanding, has_ev=True, has_pe=False)
    Chance_ev = crv_data["Chance_ev"]
    Risk_ev = crv_data["Risk_ev"]

    Expected_Value = P_up * Chance_ev - P_down * Risk_ev
    Upside_adjusted = P_up * Chance_ev
    Downside_adjusted = P_down * Risk_ev

    CRV_adjusted = Upside_adjusted / Downside_adjusted if Downside_adjusted > 0 else float('inf')

    return {
        "P_up": P_up,
        "P_down": P_down,
        "Expected_Value": Expected_Value,
        "Upside_adjusted": Upside_adjusted,
        "Downside_adjusted": Downside_adjusted,
        "CRV_adjusted": CRV_adjusted
    }


def probability_adjusted_pe(pe_series, current_pe_ratio, price, pe_ratio_list, ebitda, net_debt, eps, shares_outstanding):
    """
    Calculate probability-adjusted Risk/Reward Ratio (CRV) based on PE Ratio.

    Parameters
    ----------
    pe_series : pd.Series
        Historical PE Ratio values.
    current_pe_ratio : float
        Current PE Ratio of the company.
    price : float
        Current stock price.
    pe_ratio_list : list of float
        Historical yearly PE Ratio values, sorted.
    ebitda : float
        Current EBITDA value.
    net_debt : float
        Net debt (total debt minus cash).
    eps : float
        Earnings per share.
    shares_outstanding : int
        Number of shares outstanding.

    Returns
    -------
    dict
        Contains probability-adjusted CRV calculations:
        - "P_up": probability of upside
        - "P_down": probability of downside
        - "Expected_Value": expected monetary value
        - "Upside_adjusted": upside weighted by probability
        - "Downside_adjusted": downside weighted by probability
        - "CRV_adjusted": probability-adjusted Risk/Reward Ratio
    """
    
    hist_pe = pe_histogram(pe_series, bins=15)

    P_down = hist_pe.loc[hist_pe["High"] <= current_pe_ratio, "% of Days"].sum() / 100
    P_up = hist_pe.loc[hist_pe["Low"] > current_pe_ratio, "% of Days"].sum() / 100

    crv_data = calculate_crv(price, None, pe_ratio_list, ebitda, net_debt, eps, shares_outstanding, has_ev=False, has_pe=True)
    Chance_pe = crv_data["Chance_pe"]
    Risk_pe = crv_data["Risk_pe"]

    Expected_Value = P_up * Chance_pe - P_down * Risk_pe
    Upside_adjusted = P_up * Chance_pe
    Downside_adjusted = P_down * Risk_pe

    CRV_adjusted = Upside_adjusted / Downside_adjusted if Downside_adjusted > 0 else float('inf')

    return {
        "P_up": P_up,
        "P_down": P_down,
        "Expected_Value": Expected_Value,
        "Upside_adjusted": Upside_adjusted,
        "Downside_adjusted": Downside_adjusted,
        "CRV_adjusted": CRV_adjusted
    }
