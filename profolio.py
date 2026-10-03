from data.download_data import download_raw_data,download_features_data
from data.load_data import filter_tickers,load_features_data
from data.preprocess import cross_sectional_norm,forward_split,flatten,group_by_stock
from model.lightGBM import train_lightgbm_model
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import yfinance as yf

"choose strategy/model -> get data -> preprocess -> model -> attain output -> result"

tickers = ['A', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET', 'AON', 'AOS', 'APA', 'APD', 'APH', 'APO', 'APP', 'APTV', 'ARE', 'ARES', 'ATO', 'AVGO', 'AVY', 'AWK', 'AXON', 'AXP', 'AZO', 'BA', 'BAC', 'BALL', 'BAX', 'BBY', 'BDX', 'BEN', 'BF-B', 'BG', 'BIIB', 'BKNG', 'BKR','BLDR', 'BLK', 'BMY', 'BNY', 'BR', 'BRK-B', 'BRO', 'BSX', 'BX', 'BXP', 'C', 'CAH', 'CARR', 'CASY', 'CAT', 'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDNS', 'CDW', 'CEG', 'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CIEN', 'CINF', 'CL', 'CLX', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COHR', 'COIN', 'COO', 'COP', 'COR', 'COST', 'CPAY', 'CPRT', 'CPT', 'CRH', 'CRL', 'CRM', 'CRWD', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTSH', 'CTVA', 'CVNA', 'CVS', 'CVX', 'D', 'DAL', 'DASH', 'DD', 'DDOG', 'DE', 'DECK', 'DELL', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DLR', 'DLTR', 'DOC', 'DOV', 'DOW', 'DPZ', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXCM', 'EBAY', 'ECHO', 'ECL', 'ED', 'EFX', 'EG', 'EIX', 'EL', 'ELV', 'EME', 'EMR', 'EOG', 'EQIX', 'EQT', 'ERIE', 'ES', 'ESS', 'ETN', 'ETR', 'EVRG', 'EW', 'EXC', 'EXE', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'FCX', 'FDS', 'FDX', 'FDXF', 'FE', 'FERG', 'FFIV', 'FICO', 'FIS','FISV', 'FITB', 'FIX', 'FLEX', 'FOX', 'FOXA', 'FRT', 'FSLR', 'FTNT', 'FTV', 'GD', 'GDDY', 'GE', 'GEHC', 'GEN', 'GEV', 'GILD', 'GIS', 'GL', 'GLW', 'GM', 'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS', 'HBAN', 'HCA', 'HD', 'HIG', 'HII', 'HLT', 'HON', 'HONA', 'HOOD', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUBB', 'HUM', 'HWM', 'IBKR', 'IBM', 'ICE', 'IDXX', 'IEX', 'IFF', 'INCY', 'INTC', 'INTU', 'INVH', 'IP', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JBL', 'JCI', 'JKHY', 'JNJ', 'JPM', 'KDP', 'KEY', 'KEYS', 'KHC', 'KIM', 'KKR', 'KLAC', 'KMB', 'KMI', 'KO', 'KR', 'KVUE', 'L', 'LDOS', 'LEN', 'LH', 'LHX', 'LII', 'LIN', 'LITE', 'LLY', 'LMT', 'LNT', 'LOW', 'LRCX', 'LULU', 'LUV', 'LVS', 'LYB', 'LYV', 'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'META', 'MGM', 'MKC', 'MLM', 'MMM', 'MNST', 'MO', 'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MRSH', 'MRVL', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTB', 'MTD', 'MU', 'NCLH', 'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NOW', 'NRG','NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR', 'NWS', 'NWSA', 'NXPI', 'O', 'ODFL', 'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PANW', 'PAYX', 'PCAR', 'PCG', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PLD', 'PLTR', 'PM', 'PNC', 'PNR', 'PNW', 'PODD', 'PPG', 'PPL', 'PRU', 'PSA','PSKY', 'PSX', 'PTC', 'PWR', 'PYPL', 'Q', 'QCOM', 'RCL', 'RDDT', 'REG', 'REGN', 'RF', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'RVTY', 'SBAC', 'SBUX', 'SCHW', 'SHW', 'SJM', 'SLB', 'SMCI', 'SNA', 'SNDK', 'SNPS', 'SO', 'SOLV', 'SPG', 'SPGI', 'SRE', 'STE', 'STLD', 'STT', 'STX', 'STZ', 'SW', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP', 'TDG', 'TDY', 'TECH', 'TEL', 'TER', 'TFC', 'TGT', 'TJX', 'TKO', 'TMO', 'TMUS', 'TPL', 'TPR', 'TRGP', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TSN', 'TT', 'TTD', 'TTWO', 'TXN', 'TXT', 'TYL', 'UAL', 'UBER', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNP', 'UPS', 'URI', 'USB', 'V', 'VEEV', 'VICI', 'VLO', 'VLTO', 'VMC', 'VMRK', 'VRSK', 'VRSN', 'VRT', 'VRTX', 'VST', 'VTR', 'VTRS', 'VZ', 'WAB', 'WAT', 'WBD', 'WDAY', 'WDC', 'WEC', 'WELL', 'WFC', 'WM', 'WMB', 'WMT', 'WRB', 'WSM', 'WST', 'WTW', 'WY', 'WYNN', 'XEL', 'XOM', 'XYL', 'XYZ', 'YUM', 'ZBH', 'ZBRA', 'ZTS']
custom_features = ["RESI5", "WVMA5", "RSQR5", "KLEN", "RSQR10", "CORR5", "CORD5", "CORR10", "ROC60", "RESI10", "VSTD5", "RSQR60", "CORR60", "WVMA60", "STD5", "RSQR20", "CORD60", "CORD10", "CORR20", "KLOW"]
cross_norm = False
split_ratio = 0.8
model = "lightgbm"


#download data
download_raw_data(tickers=tickers)
download_features_data(tickers=tickers,custom_features=custom_features)

#load data
tickers = filter_tickers(tickers=tickers)
original_data = load_features_data(tickers=tickers)

#preprocess
if cross_norm:
    data = cross_sectional_norm(data=original_data)
else:
    data = original_data
train_data, valid_data = forward_split(data=data,split_ratio=split_ratio)
_, original_valid_data = forward_split(data=original_data,split_ratio=split_ratio)
train_data = flatten(train_data)
X_train,Y_train = train_data[:,:-1],train_data[:,-1]

#train model
if model == "lightgbm":
    model = train_lightgbm_model(X_train=X_train,Y_train=Y_train)

#Profolio backtesting, start from row 1895 of data(2024-10-08 to 2026-09-01)
start = "2024-10-08"
end = "2026-09-01"
daily_earn = 1
baseline = 1
random_ = 1
prof = []
base = []
rand = []

k = 50
ticker = "^GSPC"
stock = yf.Ticker(ticker)
df = stock.history(interval="1d",start=start,end=end)
close = df["Close"].values

TRADING_FEE_RATE = 0.00038
n_days = valid_data.shape[1]
total_ic = []

for i in range(n_days):
    x_valid = valid_data[:,i,:-1]
    y_pred = model.predict(x_valid)

    top_k = np.argsort(y_pred)[-k:]
    random_k = np.random.choice(len(y_pred), size=k, replace=False)
    y_real = original_valid_data[:,i,-1]
    
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

plt.clf()
plt.plot(prof,"b",label="My model selecting top-k stock")
plt.plot(rand,"r",label="Randomly choosing k stocks")
plt.plot(base,'g',label="S&P500 Baseline(Buy and hold)")
plt.legend()
plt.title(f"Profolio earning rate with transaction fee(cross-sec-norm={cross_norm})")
plt.xlabel(f"Trading day({start} to {end})")
plt.ylabel("Profolio earning rate")
fig_path = f'./result/profolio/k={k}_cross-norm={cross_norm}.png'
plt.savefig(fig_path)
print(f"profolio successfully saved to {fig_path}")

#Qauntify the result
prof_returns = np.diff(prof) / prof[:-1] # 每天的策略收益率
base_returns = np.diff(base) / base[:-1] # 每天的大盤收益率

# 1. 年化收益率 (Annualized Return)
n_years = n_days / 252.0
ann_return = (prof[-1] / prof[0]) ** (1 / n_years) - 1
base_return = (base[-1] / base[0]) ** (1/n_years) - 1

# 2. 年化夏普比率 (Annualized Sharpe Ratio)
# 每日夏普轉年化需要乘以 sqrt(252)
sharpe_ratio = (np.mean(prof_returns) / np.std(prof_returns)) * np.sqrt(252)

# 3. 最大回撤 (Maximum Drawdown)
prof_array = np.array(prof)
cum_max = np.maximum.accumulate(prof_array)
drawdowns = (prof_array - cum_max) / cum_max
max_dd = np.min(drawdowns)

# 4. Rank ICIR
ic_array = np.array(total_ic)
ic_mean = np.mean(ic_array)
icir = np.mean(ic_array) / np.std(ic_array)

#calculate 95% sharpe ratio
se_sharpe = np.sqrt((252 / n_days) * (1 + (sharpe_ratio ** 2 / 252)))
z_value = 1.96
ci_lower = sharpe_ratio - z_value * se_sharpe
ci_upper = sharpe_ratio + z_value * se_sharpe

output_filename = f"./result/quantitative/k={k}_cross-norm={cross_norm}.txt"

with open(output_filename, "w", encoding="utf-8") as f:
    f.write("\n" + "="*30 + " QUANT PERFORMANCE REPORT " + "="*30 + "\n")
    f.write(f"Annualized Return    : {ann_return * 100:.2f}%\n")
    f.write(f"Base ann return      : {base_return * 100:.2f}%\n")
    f.write(f"Annualized Sharpe    : {sharpe_ratio:.2f}\n")
    f.write(f"Maximum Drawdown     : {max_dd * 100:.2f}%\n")
    f.write(f"Mean IC              : {ic_mean:.4f}\n")
    f.write(f"ICIR                 : {icir:.2f}\n")
    f.write(f"Positive IC days     : {np.mean(ic_array > 0) * 100:.2f}%\n")
    f.write(f"Sharpe Ratio         : {sharpe_ratio:.2f}\n")
    f.write(f"Sharpe Ratio 95% CI  : [{ci_lower:.2f}, {ci_upper:.2f}]\n")
    f.write("="*86 + "\n")

print(f"Report successfully saved to {output_filename}")

