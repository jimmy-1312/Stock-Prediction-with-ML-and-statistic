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
Change LSTM structure tolearn the complex pattern.