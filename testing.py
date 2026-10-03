import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import yfinance as yf
import os

# df = pd.read_csv("./save/features_20/NVDA_features.csv",index_col=0)
# col_list = ["RESI5", "WVMA5", "RSQR5", "KLEN", "RSQR10", "CORR5", "CORD5", "CORR10", 
#             "ROC60", "RESI10", "VSTD5", "RSQR60", "CORR60", "WVMA60", "STD5", 
#             "RSQR20", "CORD60", "CORD10", "CORR20", "KLOW", "Return"
# ]
# plt.figure()
# plt.hist(df["Return"],bins=100)
# # plt.figure()
# # plt.hist(df["KLOW"]**0.25,bins=100)
# plt.show()

tickers = ['A', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALL', 'ALLE', 'AMAT', 'AMD', 'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET', 'AON', 'AOS', 'APA', 'APD', 'APH', 'APO', 'APP', 'APTV', 'ARE', 'ARES', 'ATO', 'AVGO', 'AVY', 'AWK', 'AXON', 'AXP', 'AZO', 'BA', 'BAC', 'BALL', 'BAX', 'BBY', 'BDX', 'BEN', 'BF-B', 'BG', 'BIIB', 'BKNG', 'BKR','BLDR', 'BLK', 'BMY', 'BNY', 'BR', 'BRK-B', 'BRO', 'BSX', 'BX', 'BXP', 'C', 'CAH', 'CARR', 'CASY', 'CAT', 'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDNS', 'CDW', 'CEG', 'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CIEN', 'CINF', 'CL', 'CLX', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COHR', 'COIN', 'COO', 'COP', 'COR', 'COST', 'CPAY', 'CPRT', 'CPT', 'CRH', 'CRL', 'CRM', 'CRWD', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTSH', 'CTVA', 'CVNA', 'CVS', 'CVX', 'D', 'DAL', 'DASH', 'DD', 'DDOG', 'DE', 'DECK', 'DELL', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DLR', 'DLTR', 'DOC', 'DOV', 'DOW', 'DPZ', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXCM', 'EBAY', 'ECHO', 'ECL', 'ED', 'EFX', 'EG', 'EIX', 'EL', 'ELV', 'EME', 'EMR', 'EOG', 'EQIX', 'EQT', 'ERIE', 'ES', 'ESS', 'ETN', 'ETR', 'EVRG', 'EW', 'EXC', 'EXE', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'FCX', 'FDS', 'FDX', 'FDXF', 'FE', 'FFIV', 'FICO', 'FIS','FISV', 'FITB', 'FIX', 'FLEX', 'FOX', 'FOXA', 'FRT', 'FSLR', 'FTNT', 'FTV', 'GD', 'GDDY', 'GE', 'GEHC', 'GEN', 'GEV', 'GILD', 'GIS', 'GL', 'GLW', 'GM', 'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS', 'HBAN', 'HCA', 'HD', 'HIG', 'HII', 'HLT', 'HON', 'HONA', 'HOOD', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUBB', 'HUM', 'HWM', 'IBKR', 'IBM', 'ICE', 'IDXX', 'IEX', 'IFF', 'INCY', 'INTC', 'INTU', 'INVH', 'IP', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JBL', 'JCI', 'JKHY', 'JNJ', 'JPM', 'KDP', 'KEY', 'KEYS', 'KHC', 'KIM', 'KKR', 'KLAC', 'KMB', 'KMI', 'KO', 'KR', 'KVUE', 'L', 'LDOS', 'LEN', 'LH', 'LHX', 'LII', 'LIN', 'LITE', 'LLY', 'LMT', 'LNT', 'LOW', 'LRCX', 'LULU', 'LUV', 'LVS', 'LYB', 'LYV', 'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'META', 'MGM', 'MKC', 'MLM', 'MMM', 'MNST', 'MO', 'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MRSH', 'MRVL', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTB', 'MTD', 'MU', 'NCLH', 'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NOW', 'NRG','NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR', 'NWS', 'NWSA', 'NXPI', 'O', 'ODFL', 'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PANW', 'PAYX', 'PCAR', 'PCG', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PLD', 'PLTR', 'PM', 'PNC', 'PNR', 'PNW', 'PODD', 'PPG', 'PPL', 'PRU', 'PSA','PSKY', 'PSX', 'PTC', 'PWR', 'PYPL', 'Q', 'QCOM', 'RCL', 'RDDT', 'REG', 'REGN', 'RF', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'RVTY', 'SBAC', 'SBUX', 'SCHW', 'SHW', 'SJM', 'SLB', 'SMCI', 'SNA', 'SNDK', 'SNPS', 'SO', 'SOLV', 'SPG', 'SPGI', 'SRE', 'STE', 'STLD', 'STT', 'STX', 'STZ', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP', 'TDG', 'TDY', 'TECH', 'TEL', 'TER', 'TFC', 'TGT', 'TJX', 'TKO', 'TMO', 'TMUS', 'TPL', 'TPR', 'TRGP', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TSN', 'TT', 'TTD', 'TTWO', 'TXN', 'TXT', 'TYL', 'UAL', 'UBER', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNP', 'UPS', 'URI', 'USB', 'V', 'VEEV', 'VICI', 'VLO', 'VLTO', 'VMC', 'VMRK', 'VRSK', 'VRSN', 'VRT', 'VRTX', 'VST', 'VTR', 'VTRS', 'VZ', 'WAB', 'WAT', 'WBD', 'WDAY', 'WDC', 'WEC', 'WELL', 'WFC', 'WM', 'WMB', 'WMT', 'WRB', 'WSM', 'WST', 'WTW', 'WY', 'WYNN', 'XEL', 'XOM', 'XYL', 'XYZ', 'YUM', 'ZBH', 'ZBRA', 'ZTS']
#['AMCR', 'FERG', 'SW']: error in original data
# other removed: not enough length
df = pd.read_csv(f"./save/features_20/NVDA_features.csv",index_col = 0)
index = list(df.index)
index = [date.split(" ")[0] for date in index]
col_name = list(df.columns)
# print(index)
# print(col_name)

total_x = []
total_y = []

lack = []
for ticker in tickers:
    df = pd.read_csv(f"./save/features_20/{ticker}_features.csv",index_col = 0).values

    N_DAYS = len(df)
    if N_DAYS == 2367:
        X = df[:,:-1]
        y = df[:,-1] 

        total_x.append(X)
        total_y.append(y)


#X (N_days ,N_stocks*20), y (N_days, N_stocks)
X,y = np.concat(total_x,axis=1),np.stack(total_y,axis=1)
N_stocks = y.shape[1]
X_set = []
for i in range(20):
    #shape of (N_days,N_stocks)
    X_norm = X[:,i::20]
    X_norm = (X_norm - np.mean(X_norm,axis=1).reshape(-1,1))/np.std(X_norm,axis=1).reshape(-1,1)
    X_set.append(X_norm)

yy = (y - np.mean(y,axis=1).reshape(-1,1))/np.std(y,axis=1).reshape(-1,1)
split_idx = int(N_DAYS * 0.8)

X_train_set = [X_all[:split_idx].reshape(-1) for X_all in X_set]
X_valid_set = [X_all[split_idx:].reshape(-1) for X_all in X_set]
y_train, y_valid = yy[:split_idx].reshape(-1), yy[split_idx:].reshape(-1)
#-------------------------------------
X_train = np.stack(X_train_set,axis=1)
X_valid = np.stack(X_valid_set,axis=1)


print(f"訓練集大小: {X_train.shape}")
train_data = lgb.Dataset(X_train, label=y_train)

params = {
    "objective": "regression",
    "metric": "mse",
    "learning_rate": 0.01, #0.05
    "num_leaves": 20, #20
    # "num_threads":20,
    "max_depth": 3,
    "min_data_in_leaf": 200, #200
    # "lambda_l1": 0.5,
    # "lambda_l2": 1,
    "seed": 43,
}

# callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=True)]

model = lgb.train(
    params,
    train_data,
    num_boost_round=100,           # 最大迭代次數
    # valid_sets=[train_data, valid_data], # 同時監控訓練集與測試集的 Loss
    # callbacks=callbacks
)

y_pred_valid = model.predict(X_valid)
print(y_pred_valid)

X_valid_ic_set = []
for n in range(N_stocks):
    for i in range(20):
        X_valid_ic_set.append(X_set[i][:,n])
X_valid_ic = np.stack(X_valid_ic_set,axis=1)
y_valid_ic = yy[split_idx:]

X_valid_ic = X_valid_ic[split_idx:]

total_ic = []
n_days = len(X_valid_ic)

#start from row 1895
daily_earn = 1
baseline = 1
random_ = 1
prof = []
base = []
rand = []

k = 50
ticker = "^GSPC"
stock = yf.Ticker(ticker)
df = stock.history(interval="1d",start="2024-10-08",end="2026-09-01")
close = df["Close"].values

TRADING_FEE_RATE = 0.00038

for i in range(n_days):
    x_valid = X_valid_ic[i].reshape(-1,20)
    y_pred = model.predict(x_valid)

    top_k = np.argsort(y_pred)[-k:]
    random_k = np.random.choice(len(y_pred), size=k, replace=False)
    y_real = y[split_idx:][i]
    
    model_return = np.mean(y_real[top_k])
    random_return = np.mean(y_real[random_k])

    # daily_earn *= (1+model_return)
    # random_ *= (1+random_return)
    daily_earn *= (1 + model_return) * (1 - TRADING_FEE_RATE)
    random_ *= (1 + random_return) * (1 - TRADING_FEE_RATE)
    baseline *= (close[i+1]/close[i])
    

    prof.append(daily_earn)
    base.append(baseline)
    rand.append(random_)

    ic = spearmanr(y_real, y_pred).statistic
    # ic = np.corrcoef(y_real,y_pred)[0,1]
    total_ic.append(ic)

#--------------------------------
prof_returns = np.diff(prof) / prof[:-1] # 每天的策略收益率
base_returns = np.diff(base) / base[:-1] # 每天的大盤收益率

# 1. 年化收益率 (Annualized Return)
n_years = n_days / 252.0
ann_return = (prof[-1] / prof[0]) ** (1 / n_years) - 1
base_return = (base[-1] / base[0]) ** (1/n_years) - 1

# 2. 年化夏普比率 (Annualized Sharpe Ratio, 假設無風險利率為 0)
# 每日夏普轉年化需要乘以 sqrt(252)
sharpe_ratio = (np.mean(prof_returns) / np.std(prof_returns)) * np.sqrt(252)

# 3. 最大回撤 (Maximum Drawdown)
prof_array = np.array(prof)
cum_max = np.maximum.accumulate(prof_array)
drawdowns = (prof_array - cum_max) / cum_max
max_dd = np.min(drawdowns)

# 4. Rank ICIR
rank_ic_array = np.array(total_ic) # 你的 total_ic 已經是 spearmanr 了
rank_ic_mean = np.mean(rank_ic_array)
rank_icir = np.mean(rank_ic_array) / np.std(rank_ic_array)

print("\n" + "="*30 + " QUANT PERFORMANCE REPORT " + "="*30)
print(f"Annualized Return    : {ann_return * 100:.2f}%")
print(f'Base ann return      : {base_return * 100:.2f}%')
print(f"Annualized Sharpe    : {sharpe_ratio:.2f}")
print(f"Maximum Drawdown     : {max_dd * 100:.2f}%")
print(f"Mean Rank IC         : {rank_ic_mean:.4f}")
print(f"Rank ICIR            : {rank_icir:.2f}")
se_sharpe = np.sqrt((252 / n_days) * (1 + (sharpe_ratio ** 2 / 252)))

# 2. 計算 95% 置信區間 (臨界值 Z = 1.96)
z_value = 1.96
ci_lower = sharpe_ratio - z_value * se_sharpe
ci_upper = sharpe_ratio + z_value * se_sharpe

print(f"Annualized Sharpe Ratio : {sharpe_ratio:.2f}")
print(f"Sharpe Ratio 95% CI     : [{ci_lower:.2f}, {ci_upper:.2f}]")
print("="*86)
#--------------------------------

print(f"mean ic={np.mean(total_ic)}")
print("Positive IC days:", np.mean(np.array(total_ic) > 0))
print(f'correlation betwwen y_real and y_pred{np.corrcoef(y_valid, y_pred_valid)[0, 1]}')
plt.plot(prof,"b",label="My model selecting top-k stock")
plt.plot(rand,"r",label="Randomly choosing k stocks")
plt.plot(base,'g',label="S&P500 Baseline(Buy and hold)")
plt.legend()
plt.title("Profolio earning rate with transaction fee(Applied cross-sectional norm)")
plt.xlabel("Trading day(2024-09-01 to 2026-09-01)")
plt.ylabel("Profolio earning rate")
plt.show()



