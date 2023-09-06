import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ZjhNet(nn.Module):

    def __int__(self, in_dim, out_dim):
        super(ZjhNet, self).__int__()
        self.layer1 = nn.Linear(in_dim, 100)
        self.layer2 = nn.Linear(100, 200)
        self.layer3 = nn.Linear(200, out_dim)

    def forward(self, data):
        activation1 = F.relu(self.layer1(data))
        activation2 = F.relu(self.layer2(activation1))
        return self.layer3(activation2)

