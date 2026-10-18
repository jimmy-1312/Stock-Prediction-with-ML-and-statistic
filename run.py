from data.download_data import download_raw_data,download_features_data
from data.load_data import filter_tickers,load_features_data,load_date
from data.preprocess import cross_sectional_norm,forward_split,flatten,group_by_stock
from model.lightGBM import train_lightgbm_model
from profolio.tools import profolio_simulation,save_profolio_plot
from quantitative.tools import quantitative_result,save_quantitative_result

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import yfinance as yf

"choose strategy/model -> get data -> preprocess -> model -> attain output -> result"

tickers = ['A', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET', 'AON', 'AOS', 'APA', 'APD', 'APH', 'APO', 'APP', 'APTV', 'ARE', 'ARES', 'ATO', 'AVGO', 'AVY', 'AWK', 'AXON', 'AXP', 'AZO', 'BA', 'BAC', 'BALL', 'BAX', 'BBY', 'BDX', 'BEN', 'BF-B', 'BG', 'BIIB', 'BKNG', 'BKR','BLDR', 'BLK', 'BMY', 'BNY', 'BR', 'BRK-B', 'BRO', 'BSX', 'BX', 'BXP', 'C', 'CAH', 'CARR', 'CASY', 'CAT', 'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDNS', 'CDW', 'CEG', 'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CIEN', 'CINF', 'CL', 'CLX', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COHR', 'COIN', 'COO', 'COP', 'COR', 'COST', 'CPAY', 'CPRT', 'CPT', 'CRH', 'CRL', 'CRM', 'CRWD', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTSH', 'CTVA', 'CVNA', 'CVS', 'CVX', 'D', 'DAL', 'DASH', 'DD', 'DDOG', 'DE', 'DECK', 'DELL', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DLR', 'DLTR', 'DOC', 'DOV', 'DOW', 'DPZ', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXCM', 'EBAY', 'ECHO', 'ECL', 'ED', 'EFX', 'EG', 'EIX', 'EL', 'ELV', 'EME', 'EMR', 'EOG', 'EQIX', 'EQT', 'ERIE', 'ES', 'ESS', 'ETN', 'ETR', 'EVRG', 'EW', 'EXC', 'EXE', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'FCX', 'FDS', 'FDX', 'FDXF', 'FE', 'FERG', 'FFIV', 'FICO', 'FIS','FISV', 'FITB', 'FIX', 'FLEX', 'FOX', 'FOXA', 'FRT', 'FSLR', 'FTNT', 'FTV', 'GD', 'GDDY', 'GE', 'GEHC', 'GEN', 'GEV', 'GILD', 'GIS', 'GL', 'GLW', 'GM', 'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS', 'HBAN', 'HCA', 'HD', 'HIG', 'HII', 'HLT', 'HON', 'HONA', 'HOOD', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUBB', 'HUM', 'HWM', 'IBKR', 'IBM', 'ICE', 'IDXX', 'IEX', 'IFF', 'INCY', 'INTC', 'INTU', 'INVH', 'IP', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JBL', 'JCI', 'JKHY', 'JNJ', 'JPM', 'KDP', 'KEY', 'KEYS', 'KHC', 'KIM', 'KKR', 'KLAC', 'KMB', 'KMI', 'KO', 'KR', 'KVUE', 'L', 'LDOS', 'LEN', 'LH', 'LHX', 'LII', 'LIN', 'LITE', 'LLY', 'LMT', 'LNT', 'LOW', 'LRCX', 'LULU', 'LUV', 'LVS', 'LYB', 'LYV', 'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'META', 'MGM', 'MKC', 'MLM', 'MMM', 'MNST', 'MO', 'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MRSH', 'MRVL', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTB', 'MTD', 'MU', 'NCLH', 'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NOW', 'NRG','NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR', 'NWS', 'NWSA', 'NXPI', 'O', 'ODFL', 'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PANW', 'PAYX', 'PCAR', 'PCG', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PLD', 'PLTR', 'PM', 'PNC', 'PNR', 'PNW', 'PODD', 'PPG', 'PPL', 'PRU', 'PSA','PSKY', 'PSX', 'PTC', 'PWR', 'PYPL', 'Q', 'QCOM', 'RCL', 'RDDT', 'REG', 'REGN', 'RF', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'RVTY', 'SBAC', 'SBUX', 'SCHW', 'SHW', 'SJM', 'SLB', 'SMCI', 'SNA', 'SNDK', 'SNPS', 'SO', 'SOLV', 'SPG', 'SPGI', 'SRE', 'STE', 'STLD', 'STT', 'STX', 'STZ', 'SW', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP', 'TDG', 'TDY', 'TECH', 'TEL', 'TER', 'TFC', 'TGT', 'TJX', 'TKO', 'TMO', 'TMUS', 'TPL', 'TPR', 'TRGP', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TSN', 'TT', 'TTD', 'TTWO', 'TXN', 'TXT', 'TYL', 'UAL', 'UBER', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNP', 'UPS', 'URI', 'USB', 'V', 'VEEV', 'VICI', 'VLO', 'VLTO', 'VMC', 'VMRK', 'VRSK', 'VRSN', 'VRT', 'VRTX', 'VST', 'VTR', 'VTRS', 'VZ', 'WAB', 'WAT', 'WBD', 'WDAY', 'WDC', 'WEC', 'WELL', 'WFC', 'WM', 'WMB', 'WMT', 'WRB', 'WSM', 'WST', 'WTW', 'WY', 'WYNN', 'XEL', 'XOM', 'XYL', 'XYZ', 'YUM', 'ZBH', 'ZBRA', 'ZTS']
custom_features = ["RESI5", "WVMA5", "RSQR5", "KLEN", "RSQR10", "CORR5", "CORD5", "CORR10", "ROC60", "RESI10", "VSTD5", "RSQR60", "CORR60", "WVMA60", "STD5", "RSQR20", "CORD60", "CORD10", "CORR20", "KLOW"]
cross_norm = False
split_ratio = 0.8
model_type = "lightgbm"
starting_value = 100000 #100k usd , 十萬美元
seeds = [67]
k = 50
start = "2024-10-08 00:00:00-04:00"
end = "2026-08-31 00:00:00-04:00"

#download data
download_raw_data(tickers=tickers)
download_features_data(tickers=tickers,custom_features=custom_features)

#load data
tickers = filter_tickers(tickers=tickers)
original_data = load_features_data(tickers=tickers)
date = load_date(tickers)

#preprocess
if cross_norm:
    data = cross_sectional_norm(data=original_data)
else:
    data = original_data
#(N_stock,N_train_day,N_feature),(N_stock,N_valid_day,N_feature)
train_data, valid_data = forward_split(data=data,split_ratio=split_ratio)
_, original_valid_data = forward_split(data=original_data,split_ratio=split_ratio)
valid_date = date[int(len(date)*split_ratio):]
train_data = flatten(train_data)
X_train,Y_train = train_data[:,:-1],train_data[:,-1]


ann_return_list = []
base_return_list = []
max_dd_list = []
ic_mean_list = []
postive_ic_rate_list = []
sharpe_ratio_list = []

for seed in seeds:
    #train model
    if model_type == "lightgbm":
        model = train_lightgbm_model(X_train=X_train,Y_train=Y_train,seed=seed)

    #Profolio backtesting, start from 2024-10-08 to 2026-08-31
    n_days = valid_data.shape[1]
    prof,rand,base,total_ic = profolio_simulation(
        starting_value=starting_value,
        tickers=tickers,
        model=model,
        valid_data=valid_data,
        original_valid_data=original_valid_data,
        start=start,
        end=end,
        n_days=n_days,
        k=k,
    )
    # save profolio plot
    fig_path = f"./result/profolio/k={k}_CN={'T' if cross_norm else 'F'}_seed={seed}_initial={starting_value/1000000}m.png"
    save_profolio_plot(
        prof=prof,
        rand=rand,
        base=base,
        cross_norm=cross_norm,
        start=start,
        end=end,
        fig_path=fig_path,
    )

    #Qauntify the result
    result = quantitative_result(
        prof=prof,
        rand=rand,
        base=base,
        n_days=n_days,
        total_ic=total_ic,
    )
    #Save the quantification result
    save_path = f"./result/quantitative/k={k}_CN={'T' if cross_norm else 'F'}_seed={seed}_initial={starting_value/1000000}m.txt"
    save_quantitative_result(
        save_path=save_path,
        result=result,
    )
    
    # ann_return,base_return,max_dd,ic_mean,postive_ic_rate,sharpe_ratio = result

    # ann_return_list.append(ann_return)
    # base_return_list.append(base_return)
    # max_dd_list.append(max_dd)
    # ic_mean_list.append(ic_mean)
    # postive_ic_rate_list.append(postive_ic_rate)
    # sharpe_ratio_list.append(sharpe_ratio)

    print(f"finished seed:{seed}")


# metric_lists = {
#     "Annualized Return": ann_return_list,
#     "Base Annualized Return": base_return_list,
#     "Maximum Drawdown": max_dd_list,
#     "Mean IC": ic_mean_list,
#     "Positive IC Rate": postive_ic_rate_list,
#     "Sharpe Ratio": sharpe_ratio_list,
# }

# percentage_metrics = {
#     "Annualized Return",
#     "Base Annualized Return",
#     "Maximum Drawdown",
#     "Positive IC Rate",
# }

# rows = []

# for metric_name, values in metric_lists.items():
#     values = np.asarray(values, dtype=float)
#     n = len(values)

#     if n < 2:
#         raise ValueError(
#             f"{metric_name} needs at least two values to calculate SD and 95% CI."
#         )

#     if not np.all(np.isfinite(values)):
#         raise ValueError(f"{metric_name} contains NaN or infinite values.")

#     mean = np.mean(values)
#     sd = np.std(values, ddof=1)

#     se = sd / np.sqrt(n)
#     critical_value = t.ppf(0.975, df=n - 1)
#     margin = critical_value * se

#     lower = mean - margin
#     upper = mean + margin

#     if metric_name in percentage_metrics:
#         mean_text = f"{mean * 100:.2f}%"
#         se_text = f"{se * 100:.2f}%"
#         ci_text = f"[{lower * 100:.2f}%, {upper * 100:.2f}%]"
#     elif metric_name == "Mean IC":
#         mean_text = f"{mean:.4f}"
#         se_text = f"{se:.4f}"
#         ci_text = f"[{lower:.4f}, {upper:.4f}]"
#     else:
#         mean_text = f"{mean:.2f}"
#         se_text = f"{se:.2f}"
#         ci_text = f"[{lower:.2f}, {upper:.2f}]"

#     rows.append({
#         "Metric": metric_name,
#         "Mean": mean_text,
#         "SE": se_text,
#         "95% CI": ci_text,
#     })

# summary_table = pd.DataFrame(rows)

# summary_path = (
#     f"./result/quantitative/"
#     f"summary_k={k}_CN={'T' if cross_norm else 'F'}.csv"
# )

# summary_table.to_csv(summary_path, index=False, encoding="utf-8")

# print(f"Summary table successfully saved to {summary_path}")