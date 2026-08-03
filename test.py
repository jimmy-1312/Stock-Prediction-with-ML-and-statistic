import torch
import pandas as pd
from model.LSTM import LSTM
from get_data import get_simulated_data
from train import training
import matplotlib.pyplot as plt

class LSTM_dataset(torch.utils.data.Dataset):
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

mode = "test"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size,hidden_size,num_layers = 1,50,2
epoches = 50
difficulty = "medium"
BATCH_SIZE = 100
N_days = 100

name = f"lstm_{epoches}ep_{hidden_size}hidden_{difficulty}"

if mode == "train":
    model = LSTM(input_size,hidden_size,num_layers)
    model = model.to(device)
    path = f"./save/data/{difficulty}{N_days}.csv"
    if not get_simulated_data.check_csv_data(path):
        get_simulated_data.create_csv_data(total_size=1000,N_days=N_days,variation=1,difficulty=difficulty,store_path=path)
    dataset = LSTM_dataset(path)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True)
    loss_history = training.train(model,dataloader,device,epoches)
    torch.save(model.state_dict(), f"./save/model/{name}.pth")
    plt.plot(loss_history)
    plt.savefig(f"./save/loss/{name}.png")

if mode == "test":
    model = LSTM(input_size,hidden_size,num_layers)
    model.load_state_dict(torch.load(f"./save/model/{name}.pth"))

    real_pattern = get_simulated_data.generate_pattern(100,difficulty="medium")
    plt.plot(real_pattern, 'b')

    data_with_noises = get_simulated_data.generate_data(100,1,"medium")
    data_with_noises = list(data_with_noises)[:50]
    
    with torch.no_grad():
        for i in range(100):
            out = model(torch.tensor(data_with_noises,dtype=torch.float32).view(1,-1,1))
            data_with_noises.append(out.detach().cpu().numpy()[0,-1,0])
        # out = model(torch.tensor(data_with_noises,dtype=torch.float32).view(1,-1,1))
        # out = out.detach().cpu().numpy()
    plt.plot(data_with_noises, 'r')
    # plt.plot(out[0,:,0],'g')
    # print(data_with_noises)
    # print(out)
    plt.savefig(f"./save/visualize/{name}.png")
    plt.show()
