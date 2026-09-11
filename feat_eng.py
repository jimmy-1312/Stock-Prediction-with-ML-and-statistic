import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

#deleted vwap from price feature
def Alpha157(df:pd.DataFrame, custom_features=None) -> pd.DataFrame:
    #series
    close = df["Close"]
    open_ = df["Open"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    rolling_windows = [5,10,20,30,60]
    start_pos = max(rolling_windows)
    Features = {}
    df = {}

    #----------------------------------------------
    #hard coded features * 9
    Features.update({
        "KMID": lambda:(close-open_)/open_,
        "KLEN": lambda:(high-low)/open_,
        "KMID2": lambda:(close-open_)/(high-low+1e-12),
        "KUP": lambda:(high-open_.combine(close,max))/open_,
        "KUP2": lambda:(high-open_.combine(close,max))/(high-low+1e-12),
        "KLOW": lambda:(open_.combine(close,min)-low)/open_,
        "KLOW2": lambda:(open_.combine(close,min)-low)/(high-low+1e-12),
        "KSFT": lambda:(2*close-high-low)/open_,
        "KSFT2": lambda:(2*close-high-low)/(high-low+1e-12),
    })

    #----------------------------------------------
    #price features * 3
    Features.update({
        "OPEN0": lambda:open_/close,
        "HIGH0": lambda:high/close,
        "LOW0": lambda:low/close,
    })

    #----------------------------------------------
    #rolling features * 145(29 features * 5 window size)

    #(1)ROC: Rate of Change
    Features.update({
        f"ROC{d}": lambda d=d:close.rolling(window=d,closed="both").first()/close
        for d in rolling_windows})

    #(2)MA: Moving Average
    Features.update({
        f"MA{d}": lambda d=d:close.rolling(window=d,closed="left").mean()/close
        for d in rolling_windows})

    #(3)STD: Standard deviation
    Features.update({
        f"STD{d}": lambda d=d:close.rolling(window=d,closed="left").std()/close
        for d in rolling_windows})
    
    #(4)BETA: Rate of Change
    Features.update({
        f"BETA{d}": lambda d=d:close.rolling(window=d,closed="left").apply(lambda s:(s[-1]-s[0])/(len(s)-1),raw=True)/close
        for d in rolling_windows})

    #(5)RSQR: R-square of Linear Regression --> In Simple LR, r^2 = cor(x,y)**2
    Features.update({
        f"RSQR{d}": lambda d=d:close.rolling(window=d,closed="left").apply(lambda s:np.corrcoef(s,np.arange(d))[0,1]**2,raw=True)
        for d in rolling_windows})

    #(6)RESI: Average Residual^2 for Linear Regression --> In Simple LR, SSE = SST(1 - cor(x,y)**2)
    Features.update({
        f"RESI{d}": lambda d=d:close.rolling(window=d,closed="left").apply(lambda s:np.var(s)*(1 - np.corrcoef(s,np.arange(d))[0,1]**2),raw=True)/close
        for d in rolling_windows})

    #(7)MAX: Max price of past d days
    Features.update({
        f"MAX{d}": lambda d=d:high.rolling(window=d,closed="left").max()/close
        for d in rolling_windows})

    #(8)LOW:lowest price of past d days
    Features.update({
        f"LOW{d}": lambda d=d:low.rolling(window=d,closed="left").min()/close
        for d in rolling_windows})

    #(9)QTLU: the 80% quantile of past d days
    Features.update({
        f"QTLU{d}": lambda d=d:close.rolling(window=d,closed="left").apply(lambda s:np.quantile(s,0.8),raw=True)/close
        for d in rolling_windows})

    #(10)QTLD: the 20% quantile of past d days
    Features.update({
        f"QTLD{d}": lambda d=d:close.rolling(window=d,closed="left").apply(lambda s:np.quantile(s,0.2),raw=True)/close
        for d in rolling_windows})

    #(11)RANK: the percentile of current close price in past d days
    Features.update({
        f"RANK{d}": lambda d=d:close.rolling(window=d,closed="both").apply(lambda s:np.mean(s[:-1]<=s[-1]),raw=True)
        for d in rolling_windows})

    #(12)RSV: Represent the current close price position between the history high and low in past d days
    Features.update({
        f"RSV{d}": lambda d=d:pd.DataFrame({
            "Close":close,
            "High":high,
            "Low":low,
        }).rolling(window=d,closed="both",method="table").apply(
            lambda df: [(df[-1,0]-np.min(df[:-1,2]))/(np.max(df[:-1,1])-np.min(df[:-1,2])+1e-12),np.nan,np.nan],
            raw=True,
            engine="numba")["Close"]
        for d in rolling_windows})

    #(13)IMAX: The number of days between current date and the highest price day in past d days
    Features.update({
        f"IMAX{d}": lambda d=d:high.rolling(window=d,closed="left").apply(lambda s:len(s)-np.argmax(s),raw=True)
        for d in rolling_windows})

    #(14)IMIN: The number of days between current date and the lowest price day in past d days
    Features.update({
        f"IMIN{d}": lambda d=d:low.rolling(window=d,closed="left").apply(lambda s:len(s)-np.argmin(s),raw=True)
        for d in rolling_windows})

    #(15)IMXD: The number of days between the previous highest date and the lowest date after that in past d days
    Features.update({
        f"IMXD{d}": lambda d=d:pd.DataFrame({
            "High":high,
            "Low":low,
        }).rolling(window=d,closed="left",method="table").apply(
            lambda df: [len(df) - 1 - (np.argmax(df[:, 0]) + np.argmin(df[np.argmax(df[:, 0]):, 1])), np.nan],
            raw=True,
            engine="numba")["High"]
        for d in rolling_windows})
    
    #(16)CORR: the correlation between close price and log volume in past d days
    Features.update({
        f"CORR{d}": lambda d=d:pd.DataFrame({
            "Close":close,
            "Volume":volume,
        }).rolling(window=d,closed="both",method="table").apply(
            lambda df: [np.corrcoef(df[:,0], np.log(df[:,1]+1e-12))[0,1],np.nan],
            raw=True,
            engine="numba")["Close"]
        for d in rolling_windows})

    #(17)CORD: the correlation between the close price change ratio and log volume change ratio per day, in past d days
    Features.update({
        f"CORD{d}": lambda d=d:pd.DataFrame({
            "Close":close,
            "Volume":volume,
        }).rolling(window=d,closed="both",method="table").apply(
            lambda df: [np.corrcoef(df[1:,0]/df[:-1,0], np.log(df[1:,1]/df[:-1,1]+1))[0,1],np.nan],
            raw=True,
            engine="numba")["Close"]
        for d in rolling_windows})
    
    #(18)CNTP: the percentage of days that close price goes up the next day in the past d days
    Features.update({
        f"CNTP{d}": lambda d=d:close.rolling(window=d,closed="both").apply(lambda s:np.mean((s[1:]>s[:-1])),raw=True)
        for d in rolling_windows})

    #(19)CNTN: the percentage of days that close price goes down the next day in the past d days
    Features.update({
        f"CNTN{d}": lambda d=d:close.rolling(window=d,closed="both").apply(lambda s:np.mean((s[1:]<s[:-1])),raw=True)
        for d in rolling_windows})

    #(20)CNTD: the difference between the percentage of up and down days in past d days
    Features.update({
        f"CNTD{d}": lambda d=d:close.rolling(window=d,closed="both").apply(lambda s:np.mean((s[1:]>s[:-1]))*2 - 1,raw=True)
        for d in rolling_windows})

    #(21)SUMP: the percentage of total gain inside total price change
    Features.update({
        f"SUMP{d}": lambda d=d:close.rolling(window=d,closed="both").apply(lambda s:np.sum((s[1:]-s[:-1])[s[1:]>s[:-1]])/(np.sum(np.abs(s[1:]-s[:-1]))+1e-12),raw=True)
        for d in rolling_windows})

    #(22)SUMN: the percentage of total loss inside total price change
    Features.update({
        f"SUMN{d}": lambda d=d:close.rolling(window=d,closed="both").apply(lambda s:np.sum((s[:-1]-s[1:])[s[1:]<s[:-1]])/(np.sum(np.abs(s[1:]-s[:-1]))+1e-12),raw=True)
        for d in rolling_windows})

    #(23)SUMD: the percentage diff between total gain and loss inside total price change
    Features.update({
        f"SUMD{d}": lambda d=d:close.rolling(window=d,closed="both").apply(lambda s:(np.sum((s[1:]-s[:-1])[s[1:]>s[:-1]])-np.sum((s[:-1]-s[1:])[s[1:]<s[:-1]]))/(np.sum(np.abs(s[1:]-s[:-1]))+1e-12),raw=True)
        for d in rolling_windows})

    #(24)VMA: Moving average of volume
    Features.update({
        f"VMA{d}": lambda d=d:volume.rolling(window=d,closed="both").mean()/(volume+1e-12)
        for d in rolling_windows})

    #(25)VSTD: Std of volume over past d days
    Features.update({
        f"VSTD{d}": lambda d=d:volume.rolling(window=d,closed="both").std()/(volume+1e-12)
        for d in rolling_windows})

    #(26)WVMA: Weighted volume std/mean(idk why this is useful), weighted by abs percentage change from day(t-1) to day(t)
    Features.update({
        f"WVMA{d}": lambda d=d:pd.DataFrame({
            "Close":close,
            "Volume":volume,
        }).rolling(window=d,closed="both",method="table").apply(
            lambda df: [np.std(np.abs((df[1:,0]/df[:-1,0]-1))*df[1:,1])/(np.mean(np.abs((df[1:,0]/df[:-1,0]-1))*df[1:,1])+1e-12),np.nan],
            raw=True,
            engine="numba")["Close"]
        for d in rolling_windows})
    
    #(27)VSUMP: the percentage of total gain of volume inside total volume change
    Features.update({
        f"VSUMP{d}": lambda d=d:volume.rolling(window=d,closed="both").apply(lambda s:np.sum((s[1:]-s[:-1])[s[1:]>s[:-1]])/(np.sum(np.abs(s[1:]-s[:-1]))+1e-12),raw=True)
        for d in rolling_windows})

    #(28)VSUMN: the percentage of total lose of volume inside total volume change
    Features.update({
        f"VSUMN{d}": lambda d=d:volume.rolling(window=d,closed="both").apply(lambda s:np.sum((s[:-1]-s[1:])[s[1:]<s[:-1]])/(np.sum(np.abs(s[1:]-s[:-1]))+1e-12),raw=True)
        for d in rolling_windows})

    #(29)VSUMD: the percentage diff between total gain and loss inside total volume change
    Features.update({
        f"VSUMD{d}": lambda d=d:volume.rolling(window=d,closed="both").apply(lambda s:(np.sum((s[1:]-s[:-1])[s[1:]>s[:-1]])-np.sum((s[:-1]-s[1:])[s[1:]<s[:-1]]))/(np.sum(np.abs(s[1:]-s[:-1]))+1e-12),raw=True)
        for d in rolling_windows})

    #----------------------------------------------
    #Select features
    selected_features = custom_features if custom_features else list(Features.keys())
    for feature in selected_features:
        df[feature] = Features[feature]()

    #----------------------------------------------
    #Return Target: close(t+1)/close(t) - 1
    df["Return"] = close.rolling(window=3,center=True,closed="right").apply(lambda s:s[-1]/s[-2]-1,raw=True)
    
    #----------------------------------------------
    df = pd.DataFrame(df)[start_pos:-1]
    return df

tickers = ['A', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET', 'AON', 'AOS', 'APA', 'APD', 'APH', 'APO', 'APP', 'APTV', 'ARE', 'ARES', 'ATO', 'AVGO', 'AVY', 'AWK', 'AXON', 'AXP', 'AZO', 'BA', 'BAC', 'BALL', 'BAX', 'BBY', 'BDX', 'BEN', 'BF-B', 'BG', 'BIIB', 'BKNG', 'BKR','BLDR', 'BLK', 'BMY', 'BNY', 'BR', 'BRK-B', 'BRO', 'BSX', 'BX', 'BXP', 'C', 'CAH', 'CARR', 'CASY', 'CAT', 'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDNS', 'CDW', 'CEG', 'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CIEN', 'CINF', 'CL', 'CLX', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COHR', 'COIN', 'COO', 'COP', 'COR', 'COST', 'CPAY', 'CPRT', 'CPT', 'CRH', 'CRL', 'CRM', 'CRWD', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTSH', 'CTVA', 'CVNA', 'CVS', 'CVX', 'D', 'DAL', 'DASH', 'DD', 'DDOG', 'DE', 'DECK', 'DELL', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DLR', 'DLTR', 'DOC', 'DOV', 'DOW', 'DPZ', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXCM', 'EBAY', 'ECHO', 'ECL', 'ED', 'EFX', 'EG', 'EIX', 'EL', 'ELV', 'EME', 'EMR', 'EOG', 'EQIX', 'EQT', 'ERIE', 'ES', 'ESS', 'ETN', 'ETR', 'EVRG', 'EW', 'EXC', 'EXE', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'FCX', 'FDS', 'FDX', 'FDXF', 'FE', 'FERG', 'FFIV', 'FICO', 'FIS','FISV', 'FITB', 'FIX', 'FLEX', 'FOX', 'FOXA', 'FRT', 'FSLR', 'FTNT', 'FTV', 'GD', 'GDDY', 'GE', 'GEHC', 'GEN', 'GEV', 'GILD', 'GIS', 'GL', 'GLW', 'GM', 'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS', 'HBAN', 'HCA', 'HD', 'HIG', 'HII', 'HLT', 'HON', 'HONA', 'HOOD', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUBB', 'HUM', 'HWM', 'IBKR', 'IBM', 'ICE', 'IDXX', 'IEX', 'IFF', 'INCY', 'INTC', 'INTU', 'INVH', 'IP', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JBL', 'JCI', 'JKHY', 'JNJ', 'JPM', 'KDP', 'KEY', 'KEYS', 'KHC', 'KIM', 'KKR', 'KLAC', 'KMB', 'KMI', 'KO', 'KR', 'KVUE', 'L', 'LDOS', 'LEN', 'LH', 'LHX', 'LII', 'LIN', 'LITE', 'LLY', 'LMT', 'LNT', 'LOW', 'LRCX', 'LULU', 'LUV', 'LVS', 'LYB', 'LYV', 'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'META', 'MGM', 'MKC', 'MLM', 'MMM', 'MNST', 'MO', 'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MRSH', 'MRVL', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTB', 'MTD', 'MU', 'NCLH', 'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NOW', 'NRG','NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR', 'NWS', 'NWSA', 'NXPI', 'O', 'ODFL', 'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PANW', 'PAYX', 'PCAR', 'PCG', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PLD', 'PLTR', 'PM', 'PNC', 'PNR', 'PNW', 'PODD', 'PPG', 'PPL', 'PRU', 'PSA','PSKY', 'PSX', 'PTC', 'PWR', 'PYPL', 'Q', 'QCOM', 'RCL', 'RDDT', 'REG', 'REGN', 'RF', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'RVTY', 'SBAC', 'SBUX', 'SCHW', 'SHW', 'SJM', 'SLB', 'SMCI', 'SNA', 'SNDK', 'SNPS', 'SO', 'SOLV', 'SPG', 'SPGI', 'SRE', 'STE', 'STLD', 'STT', 'STX', 'STZ', 'SW', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP', 'TDG', 'TDY', 'TECH', 'TEL', 'TER', 'TFC', 'TGT', 'TJX', 'TKO', 'TMO', 'TMUS', 'TPL', 'TPR', 'TRGP', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TSN', 'TT', 'TTD', 'TTWO', 'TXN', 'TXT', 'TYL', 'UAL', 'UBER', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNP', 'UPS', 'URI', 'USB', 'V', 'VEEV', 'VICI', 'VLO', 'VLTO', 'VMC', 'VMRK', 'VRSK', 'VRSN', 'VRT', 'VRTX', 'VST', 'VTR', 'VTRS', 'VZ', 'WAB', 'WAT', 'WBD', 'WDAY', 'WDC', 'WEC', 'WELL', 'WFC', 'WM', 'WMB', 'WMT', 'WRB', 'WSM', 'WST', 'WTW', 'WY', 'WYNN', 'XEL', 'XOM', 'XYL', 'XYZ', 'YUM', 'ZBH', 'ZBRA', 'ZTS']
with open("finished_feature.txt", "r", encoding="utf-8") as f:
    finished = {line.strip() for line in f.readlines()}

for ticker in tickers:
    if ticker not in finished:
        df = pd.read_csv(f"./save/data/{ticker}.csv",index_col=0)
        df = Alpha157(df) #["RESI5", "WVMA5", "RSQR5", "KLEN", "RSQR10", "CORR5", "CORD5", "CORR10", "ROC60", "RESI10", "VSTD5", "RSQR60", "CORR60", "WVMA60", "STD5", "RSQR20", "CORD60", "CORD10", "CORR20", "KLOW"]
        df.to_csv(f"./save/features/{ticker}_features.csv")
        print(f'{ticker} 完成！')
        with open("finished_feature.txt", "a", encoding="utf-8") as f:
            f.write(f"{ticker}\n")

# ticker = "AAPL"
# df = pd.read_csv(f"./save/data/{ticker}.csv",index_col=0)
# df = Alpha157(df,["RESI5", "WVMA5", "RSQR5", "KLEN", "RSQR10", "CORR5", "CORD5", "CORR10", 
#                             "ROC60", "RESI10", "VSTD5", "RSQR60", "CORR60", "WVMA60", "STD5", 
#                             "RSQR20", "CORD60", "CORD10", "CORR20", "KLOW"])
# df.to_csv(f"./save/features/a_{ticker}_20features_.csv")

        
# train: 1/1/2017 -> 31/12/2023
# valid: 1/1/2024 -> 31/12/2024
# test: 1/1/2025 -> 1/9/2026

#Total: 1/1/2017 -> 1/9/2026
