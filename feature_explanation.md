# Selected Feature Explanations

The model uses 20 technical features calculated from daily OHLCV data. Most
features are calculated separately for each stock and then compared across
stocks on the same day. Their usefulness is not guaranteed; the explanations
below describe possible reasons why they may help rank future stock returns.

Notation:

- $$C_t$$, $$O_t$$, $$H_t$$, and $$L_t$$ are the closing, opening, highest, and
  lowest prices on day $$t$$.
- $$V_t$$ is the trading volume.
- $$d$$ is the rolling-window length.
- $$\operatorname{Var}_d(C)$$ and $$\operatorname{Std}_d(C)$$ are rolling
  variance and standard deviation.
- $$\operatorname{Corr}_d(a,b)$$ is the rolling correlation.

## Price trend and regression features

### RESI5, RESI10

These features measure the average squared error left after fitting a linear
trend to the closing price over 5 or 10 days:

$$
\operatorname{RESI}_d =
\frac{\operatorname{Var}_d(C)
 \left(1-R_d^2\right)}{C_t}
$$

where $$R_d^2$$ is the squared correlation between closing price and time in
the rolling window.

A low value means that prices follow a relatively clear trend. A high value
means that prices fluctuate more around the trend. A low residual may indicate
strong momentum, while a high residual may indicate uncertainty or a possible
reversal. The model can learn which interpretation is more useful for the
next-day ranking.

### RSQR5, RSQR10, RSQR20, RSQR60

These features are the squared correlation between closing price and time over
5, 10, 20, and 60 days:

$$
\operatorname{RSQR}_d =
\operatorname{Corr}_d(C,\text{time})^2
$$

They measure how closely the recent price movement follows a linear trend. A
higher value indicates a more consistent upward or downward movement, although
the squared correlation does not retain the direction of the trend. These
features may help identify stocks with stable momentum or stable decline
patterns.

### ROC60

In the implementation, this feature is the first closing price in the
60-day window divided by the current closing price:

$$
\operatorname{ROC}_{60} =
\frac{C_{t-59}}{C_t}
$$

It is therefore an inverse price-change measure rather than the usual
$$C_t/C_{t-59}-1$$ definition. It gives information about the stock's longer
term price movement. Stocks with different long-term trends may have different
short-term expected returns because of momentum or mean reversion.

## Candlestick and price-range features

### KLEN

KLEN measures the daily high-low range relative to the opening price:

$$
\operatorname{KLEN}_t =
\frac{H_t-L_t}{O_t}
$$

A large value indicates a wide daily trading range and therefore stronger
intraday movement or uncertainty. It may help the model identify stocks with
high short-term activity, volatility, or breakout potential.

### KLOW

KLOW measures the distance from the lower end of the candle body to the daily
low:

$$
\operatorname{KLOW}_t =
\frac{\min(O_t,C_t)-L_t}{O_t}
$$

A large lower wick may indicate that prices fell during the day but recovered
before closing. This can reflect buying pressure or rejection of lower prices,
which may contain short-term reversal information.

## Price and volume relationship features

### CORR5, CORR10, CORR20

These features measure the rolling correlation between the closing price and
the logarithm of trading volume over 5, 10, or 20 days:

$$
\operatorname{CORR}_d =
\operatorname{Corr}_d(C,\log(V))
$$

A positive value means that higher prices tend to occur with higher volume,
while a negative value suggests the opposite. This relationship may help
distinguish price movements supported by trading activity from movements with
weaker participation.

### CORD5, CORD10, CORD60

These features measure the relationship between daily price changes and daily
volume changes:

$$
\operatorname{CORD}_d =
\operatorname{Corr}_d
\left(
\frac{C_t}{C_{t-1}},
\log\left(\frac{V_t}{V_{t-1}}+1\right)
\right)
$$

They may identify whether increases in price are accompanied by increases in
volume. A strong positive relationship could indicate confirmation of a
price trend, while a weak or negative relationship could indicate divergence
and possible instability.

### VSTD5

VSTD5 measures the standard deviation of volume over the previous 5 days
relative to the current volume:

$$
\operatorname{VSTD5} =
\frac{\operatorname{Std}_5(V)}{V_t}
$$

A high value indicates unstable or changing trading activity. Sudden volume
changes can occur around news, breakouts, or periods of stress, so this feature
may help the model identify unusual market conditions.

### WVMA5, WVMA60

WVMA measures the variability of volume weighted by the absolute daily price
change. Let

$$
w_t=\left|\frac{C_t}{C_{t-1}}-1\right|V_t.
$$

The feature is calculated as:

$$
\operatorname{WVMA}_d =
\frac{\operatorname{Std}_d(w)}{\operatorname{Mean}_d(w)}
$$

WVMA5 captures short-term changes in activity, while WVMA60 captures a
longer-term pattern. A high value may indicate irregular attention or
uncertainty. It may also help distinguish ordinary volume from volume
associated with large price movements.

## Price volatility features

### STD5

STD5 is the rolling standard deviation of closing prices over 5 days,
normalized by the current closing price:

$$
\operatorname{STD5} =
\frac{\operatorname{Std}_5(C)}{C_t}
$$

It measures short-term price volatility on a comparable scale across stocks.
Volatility may affect expected returns because high-volatility stocks can
experience larger short-term movements, but it can also indicate greater risk.
The model can learn whether high or low volatility is associated with the
next-day return in the training data.

## Summary

The selected features cover several different sources of information:

- **Trend and momentum:** ROC60 and the RSQR features;
- **Trend consistency:** the RESI features;
- **Candlestick behaviour:** KLEN and KLOW;
- **Price-volume interaction:** CORR and CORD features;
- **Trading-activity instability:** VSTD and WVMA;
- **Short-term volatility:** STD5.

These features may be effective for stock ranking because stocks do not move
independently. Relative trend strength, volatility, price patterns, and volume
participation may contain information about which stocks are more likely to
outperform others on the following day. However, the relationships can change
with market conditions, and the model's feature importance would need to be
examined before making stronger conclusions.