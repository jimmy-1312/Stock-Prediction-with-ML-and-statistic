import torch
import pandas as pd
from model.LSTM import LSTM
from get_data import get_simulated_data
from train import training
import matplotlib.pyplot as plt

class LSTM_dataset_price(torch.utils.data.Dataset):
    def __init__(self, seq_len,data_path):
        self.data = pd.read_csv(data_path,index_col=0).values
        self.length = len(self.data)
        self.seq_len = seq_len + 1
    def __len__(self):
        return self.length - self.seq_len + 1
    def __getitem__(self, idx):
        seq = self.data[idx:idx+self.seq_len]
        x_seq, y_seq = seq[:-1],seq[1:]
        return torch.tensor(x_seq,dtype=torch.float32), torch.tensor(y_seq,dtype=torch.float32)

class LSTM_dataset_logreturn(torch.utils.data.Dataset):
    def __init__(self, seq_len,data_path):
        self.data = pd.read_csv(data_path,index_col=0).values
        self.length = len(self.data)
        self.seq_len = seq_len + 1
    def __len__(self):
        return self.length - self.seq_len + 1
    def __getitem__(self, idx):
        seq = self.data[idx:idx+self.seq_len]
        x_seq, y_seq = seq[:-1],seq[1:]
        return torch.tensor(x_seq,dtype=torch.float32), torch.tensor(y_seq,dtype=torch.float32)

mode = "test"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size,hidden_size,num_layers = 1,50,1
epoches = 50
difficulty = "test"
BATCH_SIZE = 100
N_days = 1000
seq_len = 5

name = f"lstm_{epoches}ep_{hidden_size}hidden_{difficulty}"

# path = f"./save/data/{difficulty}.csv"
# dataset = LSTM_dataset_new(seq_len=5,data_path=path)
# dataloader = torch.utils.data.DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True)
# print(next(iter(dataloader))[0].shape)


if mode == "train":
    model = LSTM(input_size,hidden_size,num_layers)
    model = model.to(device)
    path = f"./save/data/{difficulty}.csv"
    if not get_simulated_data.check_csv_data(path):
        get_simulated_data.create_csv_data(N_days=N_days,variation=0,difficulty=difficulty,store_path=path)
    
    dataset = LSTM_dataset_price(seq_len=seq_len,data_path=path)
    train_dataset = LSTM_dataset_price(seq_len=seq_len,data_path=path)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True)

    loss_history = training.train(model,dataloader,device,epoches)
    torch.save(model.state_dict(), f"./save/model/{name}.pth")
    plt.plot(loss_history)
    plt.savefig(f"./save/loss/{name}.png")

if mode == "test":
    model = LSTM(input_size,hidden_size,num_layers)
    model.load_state_dict(torch.load(f"./save/model/{name}.pth"))

    real_pattern = get_simulated_data.generate_pattern(100,difficulty="test")
    plt.plot(real_pattern, 'b')

    data_with_noises = get_simulated_data.generate_data(100,0,"test")
    data_with_noises = list(data_with_noises)[:50]
    
    baseline_loss = []
    LSTM_loss = []
    with torch.no_grad():
        for i, batch in test_dataloader:
            batch = [r for r in batch]
            x, y_real = batch
            out_lstm = model(x)
    #     for i in range(100):
    #         out = model(torch.tensor(data_with_noises[-seq_len:],dtype=torch.float32).view(1,-1,1))
    #         data_with_noises.append(out.detach().cpu().numpy()[0,-1,0])
            
        # out = model(torch.tensor(data_with_noises,dtype=torch.float32).view(1,-1,1))
        # out = out.detach().cpu().numpy()
    plt.plot(data_with_noises, 'r')
    # plt.plot(out[0,:,0],'g')
    # print(data_with_noises)
    # print(out)
    plt.savefig(f"./save/visualize/{name}.png")
    plt.show()
