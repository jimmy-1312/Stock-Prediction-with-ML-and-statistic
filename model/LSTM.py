"""
To be improved: add hn and cn so when inferencing can only input the last hn and cn to get the next output
"""
import torch
import torch.nn as nn

class LSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) #shape(B,L,H)
        self.linear = nn.Linear(hidden_size,output_size) #shape(B,O)
    def forward(self, x):
        out, _ =  self.rnn(x)
        out = out[:,-1,:]
        out = self.linear(out) 
        return out