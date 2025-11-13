import torch.nn as nn
import dataset


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential()
        self.layer2 = nn.Sequential()
        self.fully_conn = nn.Linear()

    def forward(self, x):
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        h2 = h2.reshape()
        y = self.fully_conn(h2)
        return y