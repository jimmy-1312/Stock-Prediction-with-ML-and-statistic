what I wanna do:
A stock prediction model, which input is stock data, output is (1)price (2)rise or decreased** (3)vibration over a period
So the first step I wanna do is to get the data first, get them into a csv file.


Structure:

-denoising
-get data
\-get real data (from yfinance)
\-get simulated data with noises
-data storing

-model structure (in pytorch)
\-LSTM (baseline)

-training(input:model, data batches for training, real output)
-validation
-prediction(input:model, data batches for testing)


**界限**：
我到底要做什麽呢？？不是說想做一個利用模型預測股票的項目就能做的到啊，裏面要考慮的東西真的很多，包括：
(1)訓練數據間隔是多久呢（分鐘，小時，*天(先從這開始)*，年） 這關乎你這個項目能運用的範圍，（因爲用分鐘數據來訓練就只能預測分鐘）？？待驗證
(2)模型輸入是什麽，every x input is -- 單一價格，連續價格（5個）, or ? 輸出是下一個價格，上升還是下跌，confidence interval? 
(3)模型選擇

*(2) 最重要！！先從lstm 的 單一價格 x-input 來做決定*
不過先不用下載real data, 從analog data 開始(pattern + noise), 如果可以才切去real data
因爲股票本質就是有規律的數據加上noises(但是每個時刻，股票都不同distribution?)?待驗證

**待完成**
*implement "get_simulated_data"*

introduction: using this to get pattern + noises, each data point represent a date
pattern choice: -- | noises distribution choice: --

input:(N_days:how many days do you want, variation:variation of the noises)
output: data(simulate the price) in numpy format, shape of (N_days,)