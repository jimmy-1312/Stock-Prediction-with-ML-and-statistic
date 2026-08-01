"""
train_one_epoch, train, data_to_pytorch
declare model
declare training epoches, loss_creteria, data,h0,c0, true data, ADam
for epoch in cpoches-> model(data)-> loss_creteria(out,true data)-> loss.backprop
(1) how to add model to gpu? and also data,h0,c0, true data
(2) do i need to add model weights into trainer(ADam)? if yes how to make it to gpu

To be improved:
(1) Make the stock_data.csv more clear(add head,remove the left most column)
"""
import sys
sys.path.append("./")

from model.LSTM import LSTM
import numpy as np
from get_data.get_simulated_data import generate_data
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import matplotlib.pyplot as plt

EPOCHES = 100
BATCH_SIZE = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# input_size = 1
# hidden_size = 10
# model = LSTM(input_size, hidden_size).to(device)



#data shape(B,L,I) in pytorch tensor
def train(model,dataloader,device):
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    criterion = nn.MSELoss()
    epoch_iterations= len(dataloader)
    loss_history = []

    for epoch in range(EPOCHES):
        
        epoch_loss = 0

        for i,batch in enumerate(dataloader):
            
            batch = [r.to(device) for r in batch]
            x, y_real = batch

            optimizer.zero_grad()

            out = model(x)

            loss = criterion(out,y_real)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss/epoch_iterations
        loss_history.append(avg_epoch_loss)
            
        print(f'epoch = {epoch}, loss = {avg_epoch_loss}')
    
    return loss_history


class LSTM_dataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.length = len(self.data)
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        x = self.data.iloc[idx,1:-1]
        y_real = self.data.iloc[idx,2:]
        x = torch.tensor(x.values,dtype=torch.float32).view(-1,1)
        y_real = torch.tensor(y_real.values,dtype=torch.float32).view(-1,1)
        return x, y_real


model = LSTM(1,10)
model = model.to(device)
dataset = LSTM_dataset("./get_data/stock_data.csv")
dataloader = torch.utils.data.DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True)
loss_history = train(model,dataloader,device)
torch.save(model.state_dict(), "./model_save/lstm2.pth")
plt.plot(loss_history)
plt.savefig("./model_save/lstm2.png")
