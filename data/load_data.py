import pandas as pd
import numpy as np

def filter_tickers(tickers):
    """
    Check if the stock data have either one of:
    (1)missing data or zero value data (2)non-full trading dates rows
    
    Then the stock will be removed from the valid stocks list and will not be used in training
    """
    length_set = []
    missing = []

    for ticker in tickers:
        df = pd.read_csv(f"./save/data/{ticker}.csv",index_col=0)
        df = df.replace(0,np.nan)
        
        if df.isna().any().any():
            missing.append(ticker)
        
        length_set.append(len(df))

    full_length = max(length_set)
    non_full = [tickers[i] for i,length in enumerate(length_set) if length!=full_length]
    
    remaining = [ticker for ticker in tickers if (ticker not in missing) and (ticker not in non_full)]
    print(f"過濾完成，刪除了{len(missing)}個不完整數據和{len(non_full)}個交易日不夠的股票")
    print(f"Valid stock data remaining:{len(remaining)}")

    return remaining

def load_features_data(tickers):
    """
    [Return]
        data:shape of (N_stocks,N_days,N_features)
    """
    stock_list = []

    for ticker in tickers:
        df = pd.read_csv(f"./save/features_20/{ticker}_features.csv",index_col = 0).values

        stock_list.append(df)

    data = np.stack(stock_list,axis=0)
    return data
