import torch
import pandas as pd
from model.LSTM import LSTM
from model.MLP import MLP
# from preprocess import transform
from train import training
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

class LSTM_dataset(torch.utils.data.Dataset):
    def __init__(self,seq_len ,data_path):
        self.df = pd.read_csv(data_path,index_col=0).values
        # self.df = transform(self.df)
        self.len = len(self.df)
        self.seq_len = seq_len
    def __len__(self):
        return self.len - (self.seq_len) + 1
    def __getitem__(self, idx):
        seq = self.df[idx:idx+self.seq_len]
        x, y_real = seq[:self.seq_len,:-1], seq[-1][[-1]] #x:(20,20), y:(1,)
        return torch.tensor(x,dtype=torch.float32), torch.tensor(y_real,dtype=torch.float32)

class MLP_dataset(torch.utils.data.Dataset):
    def __init__(self ,data_path):
        self.df = pd.read_csv(data_path,index_col=0).values
        # self.df = transform(self.df)
        self.len = len(self.df)
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        row = self.df[idx]
        x, y_real = row[:-1], row[[-1]] #x:(20), y:(1,)
        return torch.tensor(x,dtype=torch.float32), torch.tensor(y_real,dtype=torch.float32)

class rise_stay_drop(torch.utils.data.Dataset):
    def __init__(self,input_len, pred_len ,data_path):
        self.df = pd.read_csv(data_path,index_col=0)
        self.df = feature_transform(self.df)
        self.quantile = np.quantile(self.df[:,0],q=[1/3,2/3])
        self.len = len(self.df)
        self.input_len = input_len
        self.pred_len = pred_len
    def __len__(self):
        return self.len - (self.input_len + self.pred_len) + 1
    def __getitem__(self, idx):
        seq = self.df[idx:idx+self.input_len+self.pred_len]
        x, y_real = seq[:self.input_len], seq[-1][0] #x:(10,5), y:()
        y_real = np.searchsorted(self.quantile,y_real) #to int
        return torch.tensor(x,dtype=torch.float32), torch.tensor(y_real,dtype=torch.int64)

#if flat it seems the minimum loss is 1.21, with a little bit increase
mode = "test"
lstm_mode = "value" #"value" / "category"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size,hidden_size,output_size,num_layers = 157,20,1,3
epoches = 100
BATCH_SIZE = 100
seq_len = 20

name = f"mlp_{epoches}ep_{hidden_size}hidden"
# tickers = ["NVDA"]
tickers = ['A', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE']

# dataset = Log_return(input_len=input_len,pred_len=pred_len,data_path=path)
# dataloader = torch.utils.data.DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True)
# print(next(iter(dataloader))[0][1])



if mode == "train":
    # model = LSTM(input_size,hidden_size,output_size,num_layers)
    model = MLP(input_size,output_size)
    model = model.to(device)

    dataloaders = []

    for ticker in tickers:
        path = f'./save/features/{ticker}_features.csv'
        if lstm_mode == "value":
            dataset = MLP_dataset(data_path=path)
        elif lstm_mode == "category":
            dataset = rise_stay_drop(input_len=input_len,pred_len=pred_len,data_path=path)
        dataloader = torch.utils.data.DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True)
        dataloaders.append(dataloader)

    loss_history = training.train(model,dataloaders,device,epoches,lstm_mode)
    torch.save(model.state_dict(), f"./save/model/{name}.pth")
    plt.plot(loss_history)
    plt.savefig(f"./save/loss/{name}.png")

if mode == "test":
    model = MLP(input_size,output_size)
    model.load_state_dict(torch.load(f"./save/model/{name}.pth"))

    path = f'./save/features/AAPL_features.csv'

    y_real_list = []
    y_pred_list = []
    y_baseline_list = []

    if lstm_mode == "value":
        dataset = MLP_dataset(data_path=path)

        with torch.no_grad():
            for i in range(100):
                x ,y_real = dataset[i]
                y_real_list.append(y_real.item())

                y_pred = model(x.unsqueeze(0))
                y_pred_list.append(y_pred.item())

                y_baseline_list.append(x[-1].item())
        
        y_real_list = np.array(y_real_list)
        y_pred_list = np.array(y_pred_list)
        y_baseline_list = np.array(y_baseline_list)
        lstm_rmse = np.sqrt(np.mean((y_real_list - y_pred_list)**2))
        baseline_rmse = np.sqrt(np.mean((y_real_list - y_baseline_list)**2))
        print(f"RMSE of \nLSTM : {lstm_rmse}\nBaseline : {baseline_rmse}")

        plt.plot(y_real_list, 'r', label="real")
        plt.plot(y_pred_list, 'g', label="pred")
        # plt.plot(y_baseline_list,'b',label="baseline")
        plt.legend(loc="upper right")
        plt.xlabel("Samples")
        plt.ylabel("Return Rate")
        plt.title("Return Rate Prediction")
        plt.savefig(f"./save/visualize/{name}.png")
        plt.show()

    elif lstm_mode == "category":
        dataset = rise_stay_drop(input_len=input_len,pred_len=pred_len,data_path=path)

        with torch.no_grad():
            for i in range(100):
                x ,y_real = dataset[i]
                y_real_list.append(y_real.item())

                y_pred = model(x.unsqueeze(0)).detach().cpu().numpy() #(1,3)
                y_pred = np.argmax(y_pred)
                y_pred_list.append(y_pred)

                y_baseline_list.append(2)
        
        y_real_list = np.array(y_real_list)
        y_pred_list = np.array(y_pred_list)
        y_baseline_list = np.array(y_baseline_list)

        accuracy_pred = {
            "drop": np.sum((y_real_list == y_pred_list) & (y_real_list == 0)),
            "stay": np.sum((y_real_list == y_pred_list) & (y_real_list == 1)),
            "rise": np.sum((y_real_list == y_pred_list) & (y_real_list == 2))
        }

        accuracy_base = {
            "drop": np.sum((y_real_list == y_baseline_list) & (y_real_list == 0)),
            "stay": np.sum((y_real_list == y_baseline_list) & (y_real_list == 1)),
            "rise": np.sum((y_real_list == y_baseline_list) & (y_real_list == 2))
        }

        print(confusion_matrix(y_real_list,y_pred_list))

        print(f'accuracy of predition:\n drop:{accuracy_pred["drop"]}, stay:{accuracy_pred["stay"]}, rise:{accuracy_pred["rise"]}')
        print(f'total accuracy:{np.sum(y_real_list == y_pred_list)/100}')
        print(f'accuracy of baseline:\n drop:{accuracy_base["drop"]}, stay:{accuracy_base["stay"]}, rise:{accuracy_base["rise"]}')
        print(f'total accuracy:{np.sum(y_real_list == y_baseline_list)/100}')