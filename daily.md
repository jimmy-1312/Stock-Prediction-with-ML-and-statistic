29/7/2026
初似化項目架構，目標，待完成列表
implemented the target , structures

30/7/2026
寫了一個簡易的simulated data generator, fixed pattern + fixed value + noises
next target: finish LSTM model structure, training module--> 然後先 train 一下，看一下LSTM 在noises present 的情況下去學習一個完全固定的規律，能力如何（考慮點：(1)LSTM擅長math equation的規律嗎？(2)How about when 規律一樣但是作用在不同數值上(starting point different)？(3) 規律延伸？它能摸索出equation嗎？fixed starting point but unseen data）

1/8/2026
已經8月一號早上了，昨天太晚所以先睡覺了
昨天主要完成了一個簡單的training, 并且用trained model 去在一個test data predict, 結果發現LSTM 在input 有noises 的情況下能predict the true pattern, that's because of mse loss, when data size is big enough, the prediction will converge to the mean of the y_real, this is as expected. 驚喜的點是當input 有noises 也能converge, 一個固定規律可以，但是(1)多個規律(2)更複雜的規律呢?可以是未來考慮的方向。 不過我不清楚input 有noise 的情況下去學習是不是難事，甚至可能FNN也能做的到？(input:99, output:99)need to be tested.

And at first i thought the model is unable to learn the pattern,  but later i find out it's the insufficient of training epoches, when 50->100, the later part of the pattern can be also predicted, but why it start from the left?

然後是最重要的：學到的東西
(1)
df = pd.read_csv(path) ->dataframe, df = df.values (to numpy dim=2), torch.tensor(df.values) turn numpy to tensor
pd.Dataframe(numpy.array) -> turn into dataframe -> pd.to_csv(df, PATH) -> stored csv in PATH
and actaully np.array and torch.tensor 會把elements 的 elements， 就是一個reccursion, 全部都turn into their type, it's convienient.

(2)
torch.utils.data.Dataset and Dataloader, custom dataset is so cool, it require __init__, __len__(for dataloader to know when to end),__getitem__(for dataloader to load data), super cool and convinient when loading batches.

(3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device), and make sure the input and y_real is in device as well!
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3),optimizer will track the gradient model params(if it's in gpu then track gpu directly,so make sure model.to(device) before optimizer),and also extra required by the algorithm, e.g. Adam stored the grad history as well.

(4)
torch.save(model.state_dict(),PATH) -> save the model weights
model = LSTM(), model.load(torch.load_state_dict(PATH)) -> load the weights

Next target:
(1)train a FNN for this (2)increase the difficulty (3)more pattern but still fixed starting point and length
(4)still one pattern and length but different starting point(random sample window, window size  = 50? or fewer, i saw one project is 5, is it the same idea as me) (5)test with different hidden size, lr, epoches, training data size and see the effect.


2/8/2026

剛剛才想到test data 應該是unseen data 才對，就是 >N_days 的pattern, 這個才是我們想要predict 的東西
wait.. it's not true, model may not be able to learn the unseen new pattern, although it's extend by math, but i think model cannot learn that. Instead, what it can learn is the pattern under the given system, so given exsisting datas, it can only learn the rules and connections to fulfill the whole exsisting structure, like the 0->100 days pattern, but not the new pattern derived from the existing pattern by math formula. Of course if the future pattern is included or inside the system of old data, then model can learn it, but is math equation a system? like x^2, can the model be able to learn the real rule of x^2 every where? or just learn the pattern within 0 -> 100. It's worthing to discover what the model really learnt.

Target: Discover what did the model really learn, *did it just finish the pattern, or extend the math pattern.*
And how to make the model to learn the real math behind. Is it possible with RNN or FNN.

But after a deeper thinking, i realize the new pattern after existing pattern can be actually random, it can be any curve depends on the math equation(yes?) And stock isn't math pattern, it's more likely to depends on the input data, and form one pattern, with all showing in history, there's no extend pattern.

So I am thinking of learning the method from object classification, which we can change it to classification problem, so the model output a number which represent a specific pattern(but how do we know what pattern there all--y_real), and also we can train a pattern detection model, which output the confidence of how much this would be a valid pattern(but how to know if it's valid?)

3/8/2026
[LSTM] Setting:
Data size:1000, Batch size:100, Input size:1, hidden size:50, num_layer:1 ,training epoches:50

Characteristic: 
Can be able to copy single easy pattern(one way), not able to learn vary pattern
The way it is copying instead of learning, since it only work in exisiting range, and also force the input to become the pattern no matter what the input is.

To_do:
Implement several pattern to let it learn the relationships between input and output. (Contrast learning)
Change LSTM structure to learn the complex pattern.

4/8/2026
想法（未來再看）：把目標從預測價格變成選擇行動（賣出，買入，不動）,參考behaviour cloning, conditional -> action (flow matching)
Change of strategy: seq_len = 5 , input shape = (B,L,1), output shape = (B,L,1), we only take (B,1)
To_do:
Make a baseline prediction model return (B,1) 
Make a new dataset which dataloader load (B,L,1) *done*
Make a evaluation function > input:y_pred(B,1),y_real(B,1), return RMSE

Some thought: short seq_len is better than long seq_len in rnn? with same data

RNN及普遍ML Model 强項：學習input->output 的規律，而不是predict 未來規律,或者延伸數學規律
(1)股票沒有數學規律，而且rnn 沒有能力去延伸future Pattern 
(2)rnn 需要足夠多且多變的input->output data, seq_len 太長導致training data size過少
(3)短期股票波動很大程度只depends on previous few datas, seq_len 適中即可

因此更改了：
seq_len = 5, 有多點variation的stock data, 一條stock data 即可,模仿real data,(不過可以cobine different pattern in one curve)

To_do:
Make a baseline prediction model returning the last data point
Make an evaluation function to give numeric comparision between different model
More complex pattern

8/8/2026
Choice of learning method:
online RL, offline RL, behaviour cloning

Weakness of:
--price prediction: high variation of data
--buy and sell: can it be able to understand the risk by just buy and sell data input?
--log return/trend:more stable

Change of target -> create a trading bot, time period from 1 - 4 hours(AI suggested)

I want to develop a *self-improving* trading bot! 
It is the main target, other things please don't think too much, just focus on this.

So self-improving can be achieved by:
skills(written by AI)
online RL for everyday or every week result

反正我就是想做一個agent ,可以聯通我whatsapp 每4小時去匯報一次情況，有時候不把握甚至可以問我。

New idea:
can make a pattern geneator (using classifier v.s. generator approach) for more data

下一步可以做的：
聯通whatsapp
test LSTM model ability to learn under log return
use state to art model(LightBGM,XGBoost,PatchTST)
lightbgm/xgboost suitable for predict *trend*--prob of rise|stay|drop, (3 categories)
LSTM,Amazon Chronos-2: log return

DDG-DA + TFT (Temporal Fusion Transformer)
can learn it's self learning technique


Fouund that the original y_real is wrong already, should only use the next y for target instead of a seq_y, the front x in lstm have no idea what the next y is, only if the input x is long enough and lstm learn it's pattern, it's predictable.

But i won't want to keep developing the price prediction anymore, it's time to change it to log return.

[Next]->
change the lstm structure, only predict last y *done*
change the dataset *done*
baseline(pred 0) and eval(rmse) do not need a new page, just implement inside main page, it's easy. *done*
More complex pattern *discard*
do whatsapp. *should not lsit here*

9/8/2026
why lstm can learn time series data?
what do every conversation in hermes actually pass.

11/8/2026
[Next] step:
(1)Real data input, from now on discard simulated data cuz it can not correctly simulate the character of real data. *done*
(2) LSTM-CNN
(3) transfomer
(4) Explore how each model perform, why, compare, you need to look deep into the reason, I hope to deep into the params if possible.

12/8/2026
[Next] step:
(1)Understand https://www.kaggle.com/code/kelmory/experiment-of-stock-price-prediction#Preprocessing 
for preprocessing and feature engineering
(2)See how minmax scale work, and convolution work *not useful*
(3)think about time series application ++--+ in course. *don't want*
(4)Use more data from other stocks *done*

13/8/2026
Kaggle resources from some smart guys: https://www.kaggle.com/code/kelmory/experiment-of-stock-price-prediction
data science from medium:https://medium.com/@aditib259/predicting-stock-prices-using-lstms-time-series-forecasting-a-step-by-step-guide-a70ebb04bbb8

14/8/2026
Something to change:
(1) Inside LSTM structure -- Use `Close` to select instead of number, since the close are not always in first.
(2) Inside dataset of test -- Use `Close` to select y as same reason of above.

[Next] target:
(1)more features(技術指標,指數,...pattern) __3__
(2)Add 1h interval *done*
(3)Try out XGboost or Nbeat or other quick model, to see the difference. __4__
(4)Change other 4 features to percentage difference to close. __2__ *done*
(5)Update the selector of close price, in either LSTM and test __1__ *done*

19/8/2026
These days are developing agents, today's back
Implemented a classifier dataset to see improvement(should be better since can't simply guess?)
More impression about feature engineering, 
*rule of thumb* -- features are in the same scale
To achieve this, may apply normalization, minmax scaling etc. However, not always should use normalize when data are not
following normal distribution initially, for example volume seems right skewed, so decide the scaling depends on data distribution.

[To_do]: 
(1)Diffusion model
(2)Feature engineering of volume(Now the scale is local scale, turn to global scale) **Can be dalayed**
(3)Implement a new feature(gap_indicator) 
(4)classification dataset *done*
(5)Change open,high and low(i think at least high and low should be [0,1] instead? and do they follow normal) *done*
(6)test out different ML models (file:///C:/Users/user/Downloads/Predicting_stock_returns_using_machine_learning_co.pdf)

24/8/2026
Check out qlib, see their structure and model.

25/8/2026
Today learnt:
(1) Use Decision tree(lightGBM) or light model to do feature selection on factors(qlib provide 158 factors calculating from base features)
(2) 與其學習並預測某几只股票的pattern and price(return), 不如learn the relationship between different stocks (high volume vs low colume, high price vs low price) to provide a broad view of market to model.
Advantages: a lot more of datas (規律應該是作用在所有的stocks 上面的！)
(3) finance 用詞: beta = baseline return , alpha = my_model - baseline return

[下一步]：
(1)解析qlib 的Alpha158 features engineering, 以什麽作爲target, buying rules.*done*
-> 158 features: 4 price features, 9 hard-code features, 135 rolling features, they then are being preprocessed including normalized(after appliying transformation to make them approximate to nromal), isnafill etc.
-> target: the earn from t+1 to t+2 close.
-> buying strategy: purchase the top50(preset) most earning stocks every day, sell those not in list.
(2)它是基於什麽去選股的(average return? or it's target is already average return)*done*
-> by predicted return (close(t+2)/close(t+1) -1)

it seems lightgbm is really good, lets test out it.

[論文注意]：
可復核性(讀者可以重現論文的結果)，professional quantification metric for model.

29/8/2026
既然用不了qlib, 那就由自己實現

[To_do]:
(1) Alpha158 features *done*
(2) lstm result
(3) mlp result
(4) lightgbm result
(5) Add stock percentage in S&P500 feature

30/8/2026
注意：yahoo 用了前復權(forward adjust)(keep latest price unchanged) 去解決stock split, 而qlib 用了後復權(backward adjust).
Forward adjust may cause look ahead bias, but in my case using Alpha158, the may not be a problem since alpha158 are 
capturing the percentage change of data, 無論是前復權還是後復權, 股價之間的變化幅度(以percentage計算)都是一樣.
另外，可以考慮增加一個factor 計算the percentage occupation of current stock in S&P500, 這樣就補全了Alpha158 只能捕捉變化而忽略了
股票實際價格的影響(大股票和小股票的規律可能不同)
Alpha158 的所有factor 都去除了不同股票之間股價/股份大小不一的影響, they are unit factor.

pipeline:given data period-> fetch from source -> to numpy array-> make a new feature list(length of full_len - longest window_len - target_len) -> for every feature and target -> combine into numpy array -> return numpy array

1/9/2026
change rolling output as raw for apply.

2/9/2026
Alpha157 initialized successfully, something to be aware of:
(1)implement target
(2)correlation may return nan
(3)closed "both" vs "left" inconsistent
(4)may replace .combine with pd.concat([s1,s2],axis=1).max(axis=1)

3/9/2026
[Side_Quest]
You may discover whether correlated features affect weighting of training(is it the best case that features are all independent? follow a guassian distribution with covariance = 0)
But take ur time, it's not as important as finish the project, without this you can still finish.

11/9/2026
A lot of things have been implemented: 500 stock dataset, mlp, lightgbm model, 20 selected feature v.s. 157 features on lightgbm. However, the result is not satisfied, it might scored useful R^2 for stocks including Walmart, Apple, however it can also perform extremely poor at Nvidia, google, where from the graph the model almost alway perdict 0 return, which means it learnt nothing. And with training on all 500 stocks, around 1 million data * 20 features, even if we shorten the time from 2017 -> 2026 to 2017 -> 2022, which have fewer impact by AI, the stock prediction still learnt nothing. So I'm pretty exhausted right now, what could possibly be my next step??  

23/9/2026
往後的發展方向：
基於數學模型的風險管理(如何最優化利用模型提取的alpha信號)
feature 的有效性證明, 可視化
model development
feature engineering

2/10/2026
Weekly report objective:
(1) A complete pipeline, real money,real fee, real profolio,be modulized, [data,feature,model,profolio,rules,cross-norm]
(2) Evaluation metrics
