import argparse
import torch
import torch.nn as nn
import base
import dataset

class Model(nn.module):

    def __init__(self, 
                 in_size:int, 
                 hidden_size: int,
                 rlayers: int,
                 out_size: int,
                 dropout_rate: float = 0.4):  # optimal value range is between 0.2 and 0.5
        super().__init__()
        # number of recursive layers
        self.rlayers = rlayers
        # number of hidden layers
        self.hidden_size = hidden_size 
        #
        self.dropout = nn.Dropout(p=dropout_rate)
        self.lstm = nn.LSTM(in_size,
                             hidden_size,
                             rlayers)
        self.fc = nn.Linear(hidden_size, out_size)

    def zeroes(self, x):
        # set initial state for hidden state (short-term memory)
        # and the cell state (long-term memory)
        batch_size = x.size(0)
        return torch.zeroes(self.rlayers, batch_size, self.hidden_size)
    
    def forward(self, x):
        # # initial hidden state, short-term memory
        # h0 = self.zeroes(x)
        # # initial cell state, long-term memory
        # c0 = self.zeroes(x)
        output, (hn, cn) = self.lstm(x, (self.zeroes(x), self.zeroes(x)))
        # dropout applied 
        output = self.dropout(output[:, -1, :])
        output = self.fc(output)
        return output
    

    

parser = argparse.ArgumentParser(
    prog='lstm',
    description='An LSTM model implementation for the MNIST dataset',
    epilog=''
)
parser.add_argument('-v', '--verbose', action='store_true')
base.add_hyper_params(parser)
