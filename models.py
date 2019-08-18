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
        self.out_embed = nn.Embedding(3, d_model)
        self.generator = nn.Linear(d_model, 3)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        x = self.tf(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        x = self.generator(x)
        return x
    
    def encode(self, src):
        x = self.tf.encoder(src)
        return x        

class XTransformerSim(nn.Module):

    def __init__(self, d_model=256, nhead=4, num_encoder_layers=4, dim_feedforward=1024):
        super(XTransformerSim, self).__init__()

        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward), num_encoder_layers)
        self.fc1 = nn.Linear(d_model, d_model)
        self.nl = nn.ReLU(inplace=True)
        self.ln = nn.LayerNorm(d_model)
        self.fc2 = nn.Linear(d_model, 1)
    
    def forward(self, src):
        x = self.tf(src)
        x = self.fc1(x)
        x = self.nl(x)
        x = self.ln(x)
        x = self.fc2(x)
        return x.squeeze(-1)


class XTransformerLSTMSim(nn.Module):

    def __init__(self, d_model=256, nhead=4, num_encoder_layers=2, dim_feedforward=1024):
        super(XTransformerLSTMSim, self).__init__()
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward), num_encoder_layers)
        self.decoder = AttnDecoderRNN(hidden_size = d_model)

    def forward(self, src, hidden_init):
        enc_out = self.tf(src)
        output, _, _ = self.decoder(src, hidden_init, enc_out)
        return output

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size=256, output_size=1, dropout_p=0.1, max_length=400):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        return output, hidden, attn_weights

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)


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