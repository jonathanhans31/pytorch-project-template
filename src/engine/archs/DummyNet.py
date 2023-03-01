import numpy as np 
import torch
import torch.nn as nn 
import torch.nn.functional as F

class DummyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(DummyNet, self).__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))

        for l in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
        