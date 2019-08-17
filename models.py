import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMSimilarity(nn.Module):
    
    def __init__(self, input_size=256, hidden_size=256, num_layers=2):
        super(LSTMSimilarity, self).__init__()
        self.lstm = nn.LSTM(input_size, 
                            hidden_size, 
                            num_layers=num_layers,
                            bidirectional=True)
        self.fc1 = nn.Linear(hidden_size*2, 64)
        self.nl = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc1(x)
        x = self.nl(x)
        x = self.fc2(x).squeeze(2)
        return x
