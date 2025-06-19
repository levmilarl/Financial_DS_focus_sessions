

import yfinance as yf #yfinance.download fetches multiple tickers in one shot and returns a Date×Ticker DataFrame → .pct_change() gives returns.
import pandas_datareader.data as web
from datetime import date

# 1) Asset prices ---------------------------------------------------------
tickers = ["AAPL", "MSFT", "JPM", "XOM", "KO"]
prices = yf.download(tickers, start="2010-01-01", progress=False)["Adj Close"]

# 2) Market index ---------------------------------------------------------
mkt = yf.download("^GSPC", start="2010-01-01", progress=False)["Adj Close"]

# 3) Risk-free rate (3-month T-Bill) -------------------------------------
rf = web.DataReader("DGS3MO", "fred", start="2010-01-01") / 100  # % → decimal# pandas_datareader accesses FRED (risk-free) and Ken French (FF factors); dividing by 100 converts % to decimals.


# 4) Fama-French 3-factor data -------------------------------------------
ff3 = web.DataReader("F-F_Research_Data_Factors_daily",
                     "famafrench")[0] / 100