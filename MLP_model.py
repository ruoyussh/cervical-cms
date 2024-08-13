import torch.nn as nn
import torch.nn.functional as F
import torch


class Net(nn.Module):
    def __init__(self, d, hidden_d, out_dim, dropout):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(d, d)
        self.dropout1 = nn.Dropout(dropout)
        self.hidden2 = nn.Linear(d, hidden_d)
        self.dropout2 = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_d, out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = self.hidden1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        
        x = self.hidden2(x)
        x = F.gelu(x)
        x = self.dropout2(x)

        x = self.out(x)
        x = torch.sigmoid(x)
        return x
    