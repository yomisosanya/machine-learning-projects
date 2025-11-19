import argparse
import torch
import torch.nn as nn
import base
import dataset
from tqdm import tqdm

learning_rate = 0.001

device = torch.device('cuda' if torch.cuda_is_available() else 'cpu')

class Model(nn.module):

    def __init__(self, 
                 in_size:int, 
                 hidden_size: int,
                 rlayers: int,
                 out_size: int,
                 dropout_rate: float # optimal value range is between 0.2 and 0.5
                 ):  
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
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, out_size)
        )

    def zeroes(self, x):
        # set initial state for hidden state (short-term memory)
        # and the cell state (long-term memory)
        batch_size = x.size(0)
        return torch.zeroes(self.rlayers, batch_size, self.hidden_size)
    
    def forward(self, input):
        # # initial hidden state, short-term memory
        h0 = self.zeroes(input)
        # # initial cell state, long-term memory
        c0 = self.zeroes(input)
        hidden_state_seq, (hidden_state, cell_state) = self.lstm(input, (h0, c0))
        # dropout applied to reduce the chances of overfitting 
        last_hidden_state = self.dropout(hidden_state_seq[:, -1, :])
        logits = self.classifier(last_hidden_state)
        # no activation function was applied to logits
        return logits
    
# default values
in_size = 0
hidden_size = 128
rlayers = 0
out_size = 10
dropout_rate = 0.4
    
model = Model(in_size=in_size,
              hidden_size=hidden_size,
              rlayers=rlayers,
              out_size=out_size,
              dropout_rate=dropout_rate)

    
def train_model(epochs, lr=learning_rate):
    # CrossEntropyLoss will apply an internal softmax activation function
    # to the output of the LSTM model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        
        for index, (images, labels) in enumerate(tqdm(dataset.train)):
            pass


@torch.no_grad
def test_model():
    
    for images, labels in dataset.test:
        pass
    

parser = argparse.ArgumentParser(
    prog='lstm',
    description='An LSTM model implementation for the MNIST dataset',
    epilog=''
)
parser.add_argument('-v', '--verbose', action='store_true')
base.add_hyper_params(parser)
