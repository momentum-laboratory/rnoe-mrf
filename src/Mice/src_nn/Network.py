import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, sched_iter=32):
        super(Network, self).__init__()
        dtype = torch.cuda.DoubleTensor

        self.l1 = nn.Linear(sched_iter, 300).type(dtype)
        self.relu1 = nn.ReLU().type(dtype)
        self.l2 = nn.Linear(300, 300).type(dtype)
        self.relu2 = nn.ReLU().type(dtype)
        self.l3 = nn.Linear(300, 2).type(dtype)
        self.sigmoid = nn.Sigmoid().type(dtype)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu1(x)
        x = self.l2(x)
        x = self.relu2(x)
        x = self.l3(x)
        x = self.sigmoid(x)
        return x
