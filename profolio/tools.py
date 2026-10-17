import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from profolio.profolio import Profolio
from scipy.stats import spearmanr


def profolio_simulation(starting_value,tickers,model,valid_data,original_valid_data,start,end,n_days,k):
    """
    Profolio backtesting, start from 2024-10-08 to 2026-08-31
    Args:
        starting_value: the intial amount of money invested into the profolio
        tickers,model,valid_data,original_valid_data: other needed things to construct the profolio
    Return:
        prof: A list containing the everyday networth of the profolio from the beginning to the end of backtesting, under the desired method
        rand: A list containing the everyday networth of the profolio from the beginning to the end of backtesting, under the random method
        base: A list containing the everyday networth of the profolio from the beginning to the end of backtesting, under the baseline method
        total_ic : A list containing the IC of everyday in the backtesting period
    """
    
    prof = []
    base = []
    rand = []
    
    df = pd.read_csv("save/data/^GSPC.csv",index_col=0).loc[start:end]
    baseline_close = df["Close"].values
    stock_price_list = []

    for ticker in tickers:
        df = pd.read_csv(f"save/data/{ticker}.csv",index_col=0).loc[start:end]
        df = df["Close"].values
        stock_price_list.append(df)
    stock_price = np.stack(stock_price_list,axis=0) #(N_stocks,N_valid_days+1)

    total_ic = []

    topk_prof = Profolio(tickers,starting_value)
    randk_prof = Profolio(tickers,starting_value)
    base_value = starting_value

    for i in range(n_days):
        x_valid = valid_data[:,i,:-1] #(N_stock,N_valid_day,N_feature)
        y_pred = model.predict(x_valid)

        top_k = np.argsort(y_pred)[-k:]
        random_k = np.random.choice(len(y_pred), size=k, replace=False)
        y_real = original_valid_data[:,i,-1]
        
        prof.append(topk_prof.calculate_net_worth(stock_price[:,i]))
        rand.append(randk_prof.calculate_net_worth(stock_price[:,i]))
        base.append(base_value)

        if i == 0:
            topk_prof.buy(top_k,stock_price[:,i])
            randk_prof.buy(random_k,stock_price[:,i])
        else:
            topk_prof.sell_and_buy(top_k, stock_price[:, i])
            randk_prof.sell_and_buy(random_k, stock_price[:, i])

        base_value = base_value*(baseline_close[i+1]/baseline_close[i])

        ic = spearmanr(y_real, y_pred).statistic
        total_ic.append(ic)

    prof.append(topk_prof.calculate_net_worth(stock_price[:,-1]))
    rand.append(randk_prof.calculate_net_worth(stock_price[:,-1]))
    base.append(base_value)

    return prof,rand,base,total_ic

def save_profolio_plot(prof,rand,base,cross_norm,start,end,fig_path):
    plt.clf()
    plt.plot(prof,"b",label="My model selecting top-k stock")
    plt.plot(rand,"r",label="Randomly choosing k stocks")
    plt.plot(base,'g',label="S&P500 Baseline(Buy and hold)")
    plt.legend()
    plt.title(f"Profolio earning rate with transaction fee(cross-sec-norm={cross_norm})")
    plt.xlabel(f"Trading day({start.split(' ')[0]} to {end.split(' ')[0]})")
    plt.ylabel("Profolio earning rate")
    plt.savefig(fig_path)
    print(f"profolio successfully saved to {fig_path}")