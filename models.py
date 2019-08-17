import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMSimilarity(nn.Module):

    def __init__(self, input_size=256, hidden_size=256, num_layers=2):
        super(LSTMSimilarity, self).__init__()
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers=num_layers,
                            bidirectional=True,
                            batch_first=True)
        self.fc1 = nn.Linear(hidden_size*2, 64)
        self.nl = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc1(x)
        x = self.nl(x)
        x = self.fc2(x).squeeze(2)
        return x

class pCosineSim(nn.Module):

    def __init__(self):
        super(pCosineSim, self).__init__()

    def forward(self, x):
        cs = []
        for j in range(x.shape[0]):
            cs_row = torch.clamp(torch.mm(x[j].unsqueeze(1).transpose(0,1), x.transpose(0,1)) / (torch.norm(x[j]) * torch.norm(x, dim=1)), 1e-6)
            cs.append(cs_row)
        return torch.cat(cs)

class LSTMTransform(nn.Module):

    def __init__(self, input_size=128, hidden_size=256, num_layers=2):
        super(LSTMTransform, self).__init__()
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers=num_layers,
                            bidirectional=True,
                            batch_first=True)

        self.fc1 = nn.Linear(512, 256)
        self.nl = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 256)

        self.pdist = pCosineSim()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.squeeze(0)
        x = self.fc1(x)
        x = self.nl(x)
        x = self.fc2(x)
        sim = self.pdist(x)
        return sim