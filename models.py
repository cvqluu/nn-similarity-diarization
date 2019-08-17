import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1)
    mask[mask==1.0] = float('-inf')
    return torch.FloatTensor(mask).squeeze(0)

class TransformerSim(nn.Module):

    def __init__(self, d_model=256, nhead=4, num_encoder_layers=2, dim_feedforward=1024):
        super(TransformerSim, self).__init__()

        self.tf = nn.Transformer(d_model=d_model, 
                                nhead=nhead, 
                                num_encoder_layers=num_encoder_layers,
                                num_decoder_layers=num_encoder_layers,
                                dim_feedforward=dim_feedforward)
        self.out_embed = nn.Embedding(2, d_model)
        self.generator = nn.Linear(d_model, 1)
    
    def forward(self, src, tgt, tgt_mask=None):
        x = self.tf(src, tgt, tgt_mask=tgt_mask)
        x = self.generator(x)
        return x.squeeze(2)
    
    def encode(self, src):
        x = self.tf.encoder(src)
        return x

class XTransformer(nn.Module):

    def __init__(self, d_model=128, nhead=4, num_encoder_layers=3, dim_feedforward=1024):
        super(XTransformer, self).__init__()

        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward), num_encoder_layers)
        self.fc1 = nn.Linear(d_model, d_model)
        self.nl = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(d_model, d_model)

        self.pdist = pCosineSim()
    
    def forward(self, src):
        x = self.tf(src)
        x = x.squeeze(1)
        x = self.fc1(x)
        x = self.nl(x)
        x = self.fc2(x)
        sim = self.pdist(x)
        return sim


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