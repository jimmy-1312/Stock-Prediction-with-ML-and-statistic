import yfinance as yf
import torch
import pandas as pd

apple = yf.Ticker("^GSPC")

df = apple.history(period="4y",interval="1d")

df = df.drop(["Dividends","Stock Splits"],axis=1)
df.to_csv("data.csv")


class Log_return(torch.utils.data.Dataset):
    def __init__(self,input_len, pred_len ,data_path):
        self.df = pd.read_csv("data.csv",index_col=0).values[:900]
        self.len = len(self.df)
        self.input_len = input_len
        self.pred_len = pred_len
    def __len__(self):
        return self.len - (self.input_len + self.pred_len) + 1
    def __getitem__(self, idx):
        seq = self.df[idx:idx+self.input_len+self.pred_len]
        x, y = seq[:self.input_len], seq[-1][[0]] #x:(10,5), y:(1,)
        x, y = torch.tensor(x,dtype=torch.float32), torch.tensor(y,dtype=torch.float32)
        x, y_real = torch.log(x[1:]/x[:-1]), torch.log(y/x[-1][[0]])
        return x, y_real

print(df["Volume"].values)