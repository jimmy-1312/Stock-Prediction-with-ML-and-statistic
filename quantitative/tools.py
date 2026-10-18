import numpy as np

def calculate_sharpe_ratio(return_list):
    """
    we compare the model return to the risk free method -- US bond
    Since the backtesting period start from 2024-10-08 to 2026-08-31(about 1.8945 years), we uses the interest rate from a 2-year zero-coupon bond
    delivered in October-2024, in which it's interest rate is about 4.1%
    Source: https://www.investing.com/rates-bonds/u.s.-2-year-bond-yield-historical-data
    """
    risk_free_mean_return = 1.041**(1 / 252) - 1
    sharpe_ratio = (np.mean(return_list) - risk_free_mean_return)/np.std(return_list)*np.sqrt(252)

    return sharpe_ratio
    
def quantitative_result(prof,rand,base,n_days,total_ic):
    prof_returns = np.diff(prof) / prof[:-1] # 每天的策略收益率
    base_returns = np.diff(base) / base[:-1] # 每天的大盤收益率

    # 1. 年化收益率 (Annualized Return)
    n_years = n_days / 252.0
    ann_return = (prof[-1] / prof[0]) ** (1 / n_years) - 1
    base_return = (base[-1] / base[0]) ** (1/n_years) - 1

    # 2. 年化夏普比率 (Annualized Sharpe Ratio)
    # 每日夏普轉年化需要乘以 sqrt(252)
    profolio_sharpe_ratio = calculate_sharpe_ratio(prof_returns)
    baseline_sharpe_ratio = calculate_sharpe_ratio(base_returns)

    # 3. 最大回撤 (Maximum Drawdown)
    prof_array = np.array(prof)
    cum_max = np.maximum.accumulate(prof_array)
    drawdowns = (prof_array - cum_max) / cum_max
    max_dd = np.min(drawdowns)

    # 4. Rank ICIR
    ic_array = np.array(total_ic)
    ic_mean = np.mean(ic_array)
    postive_ic_rate = np.mean(ic_array > 0)

    return (
        ann_return,
        base_return,
        max_dd,
        ic_mean,
        postive_ic_rate,
        profolio_sharpe_ratio,
        baseline_sharpe_ratio,
    )

def save_quantitative_result(save_path,result):
    ann_return,base_return,max_dd,ic_mean,postive_ic_rate,profolio_sharpe_ratio,baseline_sharpe_ratio = result
    
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n" + "="*30 + " QUANT PERFORMANCE REPORT " + "="*30 + "\n")
        f.write(f"Annualized Return    : {ann_return * 100:.2f}%\n")
        f.write(f"Base ann return      : {base_return * 100:.2f}%\n")
        f.write(f"Maximum Drawdown     : {max_dd * 100:.2f}%\n")
        f.write(f"Mean IC              : {ic_mean:.4f}\n")
        f.write(f"Positive IC days     : {postive_ic_rate * 100:.2f}%\n")
        f.write(f"Sharpe Ratio(Model)  : {profolio_sharpe_ratio:.2f}\n")
        f.write(f"Sharpe Ratio(S&P500) : {baseline_sharpe_ratio:.2f}\n")
        f.write("="*86 + "\n")

    print(f"Report successfully saved to {save_path}")