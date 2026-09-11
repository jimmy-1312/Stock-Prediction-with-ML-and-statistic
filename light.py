import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# =====================================================================
# 1. 模擬/載入你的資料 (請替換成你真實的 2000x20 特徵與 Label)
# =====================================================================
# 假設 X 是 2000x20 的 LightGBM 優選特徵，y 是 2000x1 的明天的百分比報酬率
np.random.seed(42)

# tickers = ['A', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET', 'AON', 'AOS', 'APA', 'APD', 'APH', 'APO', 'APP', 'APTV', 'ARE', 'ARES', 'ATO', 'AVGO', 'AVY', 'AWK', 'AXON', 'AXP', 'AZO', 'BA', 'BAC', 'BALL', 'BAX', 'BBY', 'BDX', 'BEN', 'BF-B', 'BG', 'BIIB', 'BKNG', 'BKR','BLDR', 'BLK', 'BMY', 'BNY', 'BR', 'BRK-B', 'BRO', 'BSX', 'BX', 'BXP', 'C', 'CAH', 'CARR', 'CASY', 'CAT', 'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDNS', 'CDW', 'CEG', 'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CIEN', 'CINF', 'CL', 'CLX', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COHR', 'COIN', 'COO', 'COP', 'COR', 'COST', 'CPAY', 'CPRT', 'CPT', 'CRH', 'CRL', 'CRM', 'CRWD', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTSH', 'CTVA', 'CVNA', 'CVS', 'CVX', 'D', 'DAL', 'DASH', 'DD', 'DDOG', 'DE', 'DECK', 'DELL', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DLR', 'DLTR', 'DOC', 'DOV', 'DOW', 'DPZ', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXCM', 'EBAY', 'ECHO', 'ECL', 'ED', 'EFX', 'EG', 'EIX', 'EL', 'ELV', 'EME', 'EMR', 'EOG', 'EQIX', 'EQT', 'ERIE', 'ES', 'ESS', 'ETN', 'ETR', 'EVRG', 'EW', 'EXC', 'EXE', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'FCX', 'FDS', 'FDX', 'FDXF', 'FE', 'FERG', 'FFIV', 'FICO', 'FIS','FISV', 'FITB', 'FIX', 'FLEX', 'FOX', 'FOXA', 'FRT', 'FSLR', 'FTNT', 'FTV', 'GD', 'GDDY', 'GE', 'GEHC', 'GEN', 'GEV', 'GILD', 'GIS', 'GL', 'GLW', 'GM', 'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS', 'HBAN', 'HCA', 'HD', 'HIG', 'HII', 'HLT', 'HON', 'HONA', 'HOOD', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUBB', 'HUM', 'HWM', 'IBKR', 'IBM', 'ICE', 'IDXX', 'IEX', 'IFF', 'INCY', 'INTC', 'INTU', 'INVH', 'IP', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JBL', 'JCI', 'JKHY', 'JNJ', 'JPM', 'KDP', 'KEY', 'KEYS', 'KHC', 'KIM', 'KKR', 'KLAC', 'KMB', 'KMI', 'KO', 'KR', 'KVUE', 'L', 'LDOS', 'LEN', 'LH', 'LHX', 'LII', 'LIN', 'LITE', 'LLY', 'LMT', 'LNT', 'LOW', 'LRCX', 'LULU', 'LUV', 'LVS', 'LYB', 'LYV', 'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'META', 'MGM', 'MKC', 'MLM', 'MMM', 'MNST', 'MO', 'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MRSH', 'MRVL', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTB', 'MTD', 'MU', 'NCLH', 'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NOW', 'NRG','NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR', 'NWS', 'NWSA', 'NXPI', 'O', 'ODFL', 'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PANW', 'PAYX', 'PCAR', 'PCG', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PLD', 'PLTR', 'PM', 'PNC', 'PNR', 'PNW', 'PODD', 'PPG', 'PPL', 'PRU', 'PSA','PSKY', 'PSX', 'PTC', 'PWR', 'PYPL', 'Q', 'QCOM', 'RCL', 'RDDT', 'REG', 'REGN', 'RF', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'RVTY', 'SBAC', 'SBUX', 'SCHW', 'SHW', 'SJM', 'SLB', 'SMCI', 'SNA', 'SNDK', 'SNPS', 'SO', 'SOLV', 'SPG', 'SPGI', 'SRE', 'STE', 'STLD', 'STT', 'STX', 'STZ', 'SW', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP', 'TDG', 'TDY', 'TECH', 'TEL', 'TER', 'TFC', 'TGT', 'TJX', 'TKO', 'TMO', 'TMUS', 'TPL', 'TPR', 'TRGP', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TSN', 'TT', 'TTD', 'TTWO', 'TXN', 'TXT', 'TYL', 'UAL', 'UBER', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNP', 'UPS', 'URI', 'USB', 'V', 'VEEV', 'VICI', 'VLO', 'VLTO', 'VMC', 'VMRK', 'VRSK', 'VRSN', 'VRT', 'VRTX', 'VST', 'VTR', 'VTRS', 'VZ', 'WAB', 'WAT', 'WBD', 'WDAY', 'WDC', 'WEC', 'WELL', 'WFC', 'WM', 'WMB', 'WMT', 'WRB', 'WSM', 'WST', 'WTW', 'WY', 'WYNN', 'XEL', 'XOM', 'XYL', 'XYZ', 'YUM', 'ZBH', 'ZBRA', 'ZTS']
# tickers = ['A', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET', 'AON', 'AOS']
tickers = ["AAPL"]
total_x_train = []
total_y_train = []
total_x_test = []
total_y_test = []
for ticker in tickers:
    df = pd.read_csv(f"./save/features/{ticker}_features.csv",index_col = 0).values

    N_DAYS = len(df)

    X_raw = df[:,:-1]
    y_raw = df[:,-1] 

    split_idx = int(N_DAYS * 0.8)

    X_train, X_test = X_raw[:split_idx], X_raw[split_idx:]
    y_train, y_test = y_raw[:split_idx], y_raw[split_idx:]
    total_x_train.append(X_train)
    total_x_test.append(X_test)
    total_y_train.append(y_train)
    total_y_test.append(y_test)

X_train,X_test = np.concat(total_x_train,axis=0),np.concat(total_x_test,axis=0)
y_train, y_test = np.concat(total_y_train,axis=0),np.concat(total_y_test,axis=0)
print(f"訓練集大小: {X_train.shape}, 測試集大小: {X_test.shape}")

# =====================================================================
# 3. 建立 LightGBM 專用資料集與超參數設定
# =====================================================================
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# plt.hist(y_train, bins=1000)
# plt.show()
params = {
    "objective": "regression",
    "metric": "mse",
    "learning_rate": 0.05,
    "num_leaves": 20,
    "max_depth": 3,
    "min_data_in_leaf": 100, #200
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 42,
}
# params = {
#     "objective": "regression",
#     "metric": "mse",
#     "learning_rate": 0.05,
#     "num_leaves": 128,
#     "max_depth": -1,
#     "min_data_in_leaf": 1,
#     "feature_fraction": 1.0,
#     "bagging_fraction": 1.0,
#     "bagging_freq": 0,
#     "lambda_l1": 0,
#     "lambda_l2": 0,
#     "verbosity": -1,
#     "seed": 42,
# }


# =====================================================================
# 4. 模型訓練 (包含 Early Stopping)
# =====================================================================
callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=True)]

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,           # 最大迭代次數
    valid_sets=[train_data, test_data], # 同時監控訓練集與測試集的 Loss
    callbacks=callbacks
)
# model = lgb.train(
#     params,
#     train_data,
#     num_boost_round=2000
# )

# =====================================================================
# 5. 預測與評估
# =====================================================================
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# 計算指標
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2 = r2_score(y_test, y_pred_test)
ic = spearmanr(y_test, y_pred_test).statistic
# baseline_pred = np.full_like(y_test, y_train.mean())
# baseline_r2 = r2_score(y_test, baseline_pred)

# zero_r2 = r2_score(y_test, np.zeros_like(y_test))

# print("\n" + "="*40)
# print(f"【訓練集 RMSE】: {train_rmse:.6f}")
# print(f"【測試集 RMSE】: {test_rmse:.6f}")
# print(f"【測試集 R² 分數】: {test_r2:.6f} (越接近 1 代表預測越準, 小於 0 代表不如猜均值)")
print("="*40)
print(f'Real sd{np.std(y_test)}')
print(f'test sd{np.std(y_pred_test)}')
print(f'correlation betwwen y_real and y_pred{np.corrcoef(y_test, y_pred_test)[0, 1]}')
print(f'ic = {ic}')
# train_r2 = r2_score(y_train, y_pred_train)
# print(f"Train R²: {train_r2:.6f}")


# =====================================================================
# 6. 視覺化檢查：看它是不是還是一條直線！
# =====================================================================
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='True Return (Actual)', color='blue', alpha=0.6)
plt.plot(y_pred_test, label='LightGBM Prediction', color='red', linestyle='--')
plt.title('Apple Stock Return Prediction (Test Set)')
plt.xlabel('Days')
plt.ylabel('Percentage Return')
plt.legend()
plt.grid(True)
plt.show()

# 印出前 10 筆預測值，親眼證實有沒有變動
# print("\n測試集前 10 筆真實值與預測值對比:")
# df_check = pd.DataFrame({'True': y_test[:10], 'Pred': y_pred_test[:10]})
# print(df_check)
