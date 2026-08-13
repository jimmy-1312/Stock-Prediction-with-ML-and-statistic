import yfinance as yf
import pandas as pd
import numpy as np

tickers = ["MSFT", "AAPL", "NVDA", "AMZN", "META", "GOOGL", "BRK-B", "LLY", "AVGO", "TSLA"]
period = "5y"
interval = "1d"

def fill_zero_and_nan(df):
    df = df.replace(0,np.nan)
    row_idx, col_idx = np.where(df.isna())

    for i,j in zip(row_idx,col_idx):
        if i == 0:
            df.iat[i,j] = df.iat[i+1,j]
        else:
            df.iat[i,j] = df.iat[i-1,j]
    
    return df

for ticker in tickers:
    
    stock = yf.Ticker(ticker)
    df = stock.history(period=period,interval=interval)

    df = df.drop(["Dividends","Stock Splits"],axis=1)
    df = fill_zero_and_nan(df)
    
    df.to_csv(f"./save/data/{ticker}_{period}.csv")
