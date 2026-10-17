import numpy as np

def quantitative_result(prof,rand,base,n_days,total_ic):
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
    postive_ic_rate = np.mean(ic_array > 0)

    return (
        ann_return,
        base_return,
        max_dd,
        ic_mean,
        postive_ic_rate,
        sharpe_ratio,
    )

def save_quantitative_result(save_path,result):
    ann_return,base_return,max_dd,ic_mean,postive_ic_rate,sharpe_ratio = result
    
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n" + "="*30 + " QUANT PERFORMANCE REPORT " + "="*30 + "\n")
        f.write(f"Annualized Return    : {ann_return * 100:.2f}%\n")
        f.write(f"Base ann return      : {base_return * 100:.2f}%\n")
        f.write(f"Maximum Drawdown     : {max_dd * 100:.2f}%\n")
        f.write(f"Mean IC              : {ic_mean:.4f}\n")
        f.write(f"Positive IC days     : {postive_ic_rate * 100:.2f}%\n")
        f.write(f"Sharpe Ratio         : {sharpe_ratio:.2f}\n")
        f.write("="*86 + "\n")

    print(f"Report successfully saved to {save_path}")