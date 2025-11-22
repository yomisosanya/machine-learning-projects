import argparse
import base
import dataset
import torch
import torch.nn as nn
from torchmetrics import Accuracy, Precision, Recall
from tqdm import tqdm
import sys

learning_rate = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):

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
                             rlayers,
                             batch_first=True
                            )
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, out_size)
        )

    def zeroes(self, x):
        # set initial state for hidden state (short-term memory)
        # and the cell state (long-term memory)
        batch_size = x.size(0)
        return torch.zeros(self.rlayers, batch_size, self.hidden_size).to(x.device)
    
    def forward(self, x):
        input = x.squeeze(1)
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
in_size = 28
hidden_size = 128
rlayers = 2
out_size = 10
dropout_rate = 0.4
verbose = True
    
model = Model(in_size=in_size,
              hidden_size=hidden_size,
              rlayers=rlayers,
              out_size=out_size,
              dropout_rate=dropout_rate)

    
def train_model(epochs=3, lr=learning_rate, out=sys.stdout):
    # CrossEntropyLoss will apply an internal softmax activation function
    # to the output of the LSTM model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    kwargs = {'task': 'multiclass', 'num_classes': out_size}
    accuracy = Accuracy(**kwargs)
    precision = Precision(average='macro', **kwargs)
    recall = Recall(average='macro', **kwargs)

    for epoch in range(epochs):
        print(f'epoch {epoch + 1} / epochs', file=out)
        for index, (images, labels) in enumerate(tqdm(dataset.train_loader)):
            probabilities = model(images)
            predictions = torch.argmax(probabilities, dim=1)
            accuracy.update(predictions, labels)
            precision.update(predictions, labels)
            recall.update(predictions, labels)
        if verbose:
            print(f'epoch accuracy: {accuracy.compute()}')
            print(f'epoch precision: {precision.compute()}')
            print(f'epoch recall: {recall.compute()}')
        accuracy.reset()
        precision.reset()
        recall.reset()


@torch.no_grad
def test_model():
    pass


    

parser = argparse.ArgumentParser(
    prog='lstm',
    description='An LSTM model implementation for the MNIST dataset',
    epilog=''
)
parser.add_argument('-v', '--verbose', action='store_true')
base.add_hyper_params(parser)



# test code
train_model()

