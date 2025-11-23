import argparse
import math
import torch
import torch.nn as nn
from torchmetrics import Accuracy, Precision, Recall
import dataset
import sys
from tqdm import tqdm
from typing import TextIO

d0 = 1  # input dimension
d1 = 10 # number of output

c1 = 64  # filter 1
c2 = 256 # filter 2
fc0 = 512
cls = 10  # number of classes

dest_path = 'cnn_model.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def maxpool_dim(h, w, *,  kernel, stride=None, padding=0, dilation=1):
    if stride is None:
        stride = kernel
    func = lambda k: ((k + 2 * padding - dilation * (kernel - 1) - 1)/stride) + 1
    return math.floor(func(h)), math.floor(func(w))
            


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.filter = nn.Sequential(
            # grayscale immage, out-channel = 64, simple feature like edgeslarge 
            # large kernel size to focus more on general shape
            # (1, 28, 28) --> (64, 28, 28)
            nn.Conv2d(in_channels=d0, out_channels=c1, kernel_size=4, padding=2), 
            nn.ReLU(),
            # (64, 28, 28) --> (64, 14, 14)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # out-channel = 256, more complex feature and details
            # (64, 14, 14) --> (128, 14, 14)
            nn.Conv2d(in_channels=c1, out_channels=c1*2, kernel_size=2, padding=1),
            nn.ReLU(),
            # (128, 14, 14) --> (128, 7, 7)
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        #
        hx, wx = maxpool_dim(14, 14, kernel=2, stride=2)

        self.classifier = nn.Sequential(
            # (128, 7, 7) --> (64, 128*7*7)
            nn.Flatten(),
            # (64, 6272) --> (6272, 512)
            nn.Linear(hx*wx*128, fc0),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            # (6272,512) --> (512, 512)
            nn.Linear(fc0, fc0),
            nn.ReLU(),
            # (512, 512) --> (512, 10)
            nn.Linear(fc0, cls)
        )

        self.predict = nn.Softmax(dim=1)

    def forward(self, input):
        features = self.filter(input)
        logits = self.classifier(features)
        probablities = self.predict(logits)
        return probablities
    
model = Model()
# loss function
criterion = nn.CrossEntropyLoss()

best_val = float('inf')



def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0  # batches that have been seen

    for (images, labels) in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)
        probabilities = model(images)
        loss = criterion(probabilities, labels)
        bs = labels.size(0) # batch size
        total_loss += loss.item() * bs
        total_acc += (probabilities.argmax(1) == labels).float().sum().item()
        n += bs
    return (total_loss / n), (total_acc / n)


def train_model(epochs, *, optimizer, verbose: bool = False):
    best_val = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n = 0  # batches that have been seen


        for (images, labels) in tqdm(dataset.train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            probabilities = model(images)
            loss = criterion(probabilities, labels)
            loss.backward()
            optimizer.step()
            bs = labels.size(0)
            running_loss += loss.item() * bs
            running_acc += (probabilities.argmax(1) == labels).float().sum().item()
            n += bs

        train_loss = running_loss / n
        train_acc = running_acc / n
        print('validation...')
        val_loss, val_acc = evaluate(dataset.validation_loader)

        print(f'Epoch {epoch:02d}')
        print(f'training loss: {train_loss:.4f}')
        print(f'validation loss: {val_loss:.4f}')
        print(f'accuracy: {val_acc:.4f}')

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), dest_path)



def test_model():
    model.load_state_dict(torch.load(dest_path, map_location=device))
    loss, acc = evaluate(dataset.test_loader)
    print(f'Test')
    print(f'loss: {loss:.4f}')
    print(f'accuracy: {acc:.4f}')



parser = argparse.ArgumentParser(
    prog='cnn',
    description='A Convolutional Neural Network (CNN)  model implementation for the MNIST dataset',
    epilog=''
)

parser.add_argument('procedure', choices=['train', 'test'], help='Choose a procedure: [train | test]')
parser.add_argument('-v', '--verbose', action='store_true')
# parser.add_argument('-s', '--save', action='store_true')

parser.add_argument('-b', '--bidirectional', action='store_true')  # lstm specific
parser.add_argument('-w', '--bias-weights', action='store_true')  # lstm specific
parser.add_argument('--lr', default=0.001, type=float, help='The learning rate of the model. The default value is 0.001')
parser.add_argument('--epoch', default=3, type=int, help='The number of passes')
parser.add_argument('--optimizer',
                    choices=['Adam', 'RMSprop', 'SGD'],
                    default='Adam',
                    help='Choose an optimizer: [ Adam | RMSprop | SGD ], the default is Adam ')


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
        train_model(args.epoch, optimizer=optimizer, verbose=args.verbose)
    case 'test':
        test_model()


