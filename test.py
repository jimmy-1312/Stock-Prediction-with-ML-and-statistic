import torch
import pandas as pd
from model.LSTM import LSTM
from get_data import get_simulated_data
from train import training
import matplotlib.pyplot as plt
import numpy as np

class Log_return(torch.utils.data.Dataset):
    def __init__(self, input_len, pred_len ,data_path):
        self.data = pd.read_csv(data_path,index_col=0).values #shape of (1000,1)
        self.length = len(self.data)
        self.input_len = input_len
        self.seq_len = input_len + pred_len
    def __len__(self):
        return self.length - self.seq_len + 1
    def __getitem__(self, idx):
        seq = self.data[idx:idx+self.seq_len]
        x, y = seq[:self.input_len], seq[-1]
        x, y = torch.tensor(x,dtype=torch.float32), torch.tensor(y,dtype=torch.float32)
        x, y_real = torch.log(x[1:]/x[:-1]), torch.log(y/x[-1])
        return x, y_real

mode = "test"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size,hidden_size,output_size,num_layers = 1,50,1,1
epoches = 50
difficulty = "test"
BATCH_SIZE = 100
N_days = 1000
input_len, pred_len = 6,4

name = f"lstm_{epoches}ep_{hidden_size}hidden_{difficulty}"

path = f"./save/data/{difficulty}.csv"
# dataset = Log_return(input_len=input_len,pred_len=pred_len,data_path=path)
# dataloader = torch.utils.data.DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True)
# print(next(iter(dataloader))[1].shape)


if mode == "train":
    model = LSTM(input_size,hidden_size,output_size,num_layers)
    model = model.to(device)
    path = f"./save/data/{difficulty}.csv"
    if not get_simulated_data.check_csv_data(path):
        get_simulated_data.create_csv_data(N_days=N_days,variation=0,difficulty=difficulty,store_path=path)
    
    dataset = Log_return(input_len=input_len,pred_len=pred_len,data_path=path)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True)

    loss_history = training.train(model,dataloader,device,epoches)
    torch.save(model.state_dict(), f"./save/model/{name}.pth")
    plt.plot(loss_history)
    plt.savefig(f"./save/loss/{name}.png")

if mode == "test":
    model = LSTM(input_size,hidden_size,output_size,num_layers)
    model.load_state_dict(torch.load(f"./save/model/{name}.pth"))

    dataset = Log_return(input_len=input_len,pred_len=pred_len,data_path=path)

    y_real_list = []
    y_pred_list = []
    y_baseline_list = []
    with torch.no_grad():
        for i in range(100):
            x ,y_real = dataset[i]
            y_real_list.append(y_real.item())

            y_pred = model(x.unsqueeze(0))
            y_pred_list.append(y_pred.item())

            y_baseline_list.append(0)
    
    # plt.plot(y_real_list, 'r', label="real")
    # plt.plot(y_pred_list, 'g', label="pred")
    # plt.plot(y_baseline_list,'b',label="baseline")
    # plt.legend(loc="upper right")
    # plt.xlabel("Samples")
    # plt.ylabel("Return Rate")
    # plt.title("Return Rate Prediction")
    # plt.savefig(f"./save/visualize/{name}.png")
    # plt.show()
    y_real_list = np.array(y_real_list)
    y_pred_list = np.array(y_pred_list)
    y_baseline_list = np.array(y_baseline_list)
    lstm_rmse = np.sqrt(np.mean((y_real_list - y_pred_list)**2))
    baseline_rmse = np.sqrt(np.mean((y_real_list - y_baseline_list)**2))
    print(f"RMSE of \nLSTM : {lstm_rmse}\nBaseline : {baseline_rmse}")