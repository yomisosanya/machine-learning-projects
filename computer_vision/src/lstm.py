import argparse
import base
import dataset
import os
import torch
import torch.nn as nn
from torchmetrics import Accuracy, Precision, Recall
from tqdm import tqdm
import shelve
import sys

learning_rate = 0.001
dest_path = 'lstm_model.pt'

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
            nn.Linear(self.hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, out_size)
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
hidden_size = 16
rlayers = 2
out_size = 10
dropout_rate = 0.4
verbose = True
    
model = Model(in_size=in_size,
              hidden_size=hidden_size,
              rlayers=rlayers,
              out_size=out_size,
              dropout_rate=dropout_rate)

criterion = nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for (images, labels) in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)
        probabilities = model(images)
        predictions = torch.argmax(probabilities, dim=1)
        total_acc += (predictions == labels).float().sum().item()
        bs = labels.size(0)
        n += bs
        loss = criterion(probabilities, labels)
        total_loss += loss.item() * bs

    return (total_loss / n), (total_acc / n)


def train_model(*, 
                epochs,
                verbose,
                lr,
                optimizer):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n = 0

        for (images, labels) in tqdm(dataset.train_loader):
            images = images.to(device)
            labels = labels.to(device)
            probabilities = model(images)
            running_acc += (probabilities.argmax(1) == labels).float().sum().item()
            loss = criterion(probabilities, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bs = labels.size(0)
            running_loss += loss.item() * bs
            n += bs

        train_loss = running_loss / n
        train_acc = running_acc / n
        print('validation...')
        val_loss, val_acc = evaluate(dataset.validation_loader)
        print(f'Epoch {epoch:02d}')
        print(f'training loss: {train_loss:.4f}')
        print(f'validation loss: {val_loss:.4f}')
        print(f'accuracy: {val_acc:.4f}')

        torch.save(model.state_dict(), dest_path)
        

    # with shelve.open('lstm.store') as f:
       # if 'loss' in f:
           # if f['loss'] > val_loss:
               # torch.save(model.state_dict(), dest_path)
               # f['loss'] = val_loss


def test_model():
    model.load_state_dict(torch.load(dest_path, map_location=device))
    loss, acc = evaluate(dataset.test_loader)
    print(f'Test')
    print(f'loss: {loss:.4f}')
    print(f'accuracy: {acc:.4f}')


    

parser = argparse.ArgumentParser(
    prog='lstm',
    description='An LSTM model implementation for the MNIST dataset',
    epilog=''
)

parser.add_argument('procedure', choices=['train', 'test'], help='Choose a procedure: [train | test]')
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('-s', '--save', action='store_true')

parser.add_argument('-b', '--bidirectional', action='store_true')  # lstm specific
parser.add_argument('-w', '--bias-weights', action='store_true')  # lstm specific
parser.add_argument('--lr', default=0.001, type=float, help='The learning rate of the model. The default value is 0.001')
parser.add_argument('--epoch', default=3, type=int, help='The number of passes')
parser.add_argument('--optimizer', 
                    choices=['Adam', 'RMSprop', 'SGD'],
                    default='Adam',
                    help='Choose an optimizer: [ Adam | RMSprop | SGD ], the default is Adam ')
# parser.add_argument('--hidden-size', default=


args = parser.parse_args()
opt = args.optimizer

match opt.casefold():
    case 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    case 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    case 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    case _:
        raise ValueError('Unsupported optimizer')


match args.procedure:
    case 'train':
        train_model(epochs=args.epoch,
                    lr=args.lr,
                    optimizer=optimizer, 
                    verbose=args.verbose)
    case 'test':
        test_model()


