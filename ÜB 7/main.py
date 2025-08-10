import yfinance as yf
import pandas as pd

tickers = ["AAPL", "MSFT", "GOOGL"]
data = yf.download(tickers=tickers, start="2025-05-01", end="2025-08-09")
print(data)