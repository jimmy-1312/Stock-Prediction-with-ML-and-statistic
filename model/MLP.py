import torch
import torch.nn as nn

class MLP(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.MLP_block = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32,output_dim)
        )

    #x:(B,I) -> out:(B,O)
    def forward(self, x):
        out =  self.MLP_block(x)
        return out