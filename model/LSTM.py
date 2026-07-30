import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        rnn = nn.LSTM(input_size,hidden_size,)