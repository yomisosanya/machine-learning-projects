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
verbose = True

c0 = 1
c1 = 64
c2 = 256
fc0 = 512
cls = 10  # number of classes

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
            nn.Flatten(),
            nn.Linear(hx*wx*128, fc0),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(fc0, fc0),
            nn.ReLU(),
            nn.Linear(fc0, cls)
        )

        self.predict = nn.Softmax(dim=1)

    def forward(self, input):
        features = self.filter(input)
        logits = self.classifier(features)
        probablities = self.predict(logits)
        return probablities
    
forward_pass = Model()


# loss function
criterion = nn.CrossEntropyLoss()


def train_model(epochs: int = 3, *, verbose: bool = False, out: TextIO = sys.stdout) :

    kwargs = {'task': 'multiclass', 'num_classes': cls}
    accuracy = Accuracy(**kwargs)
    precision = Precision(average='macro', **kwargs)
    recall = Recall(average='macro', **kwargs)
    if verbose:
        pass

    for epoch in range(epochs):
        print(f'epoch {epoch + 1}/{epochs}', file=out)
        for index, (images, labels) in enumerate(tqdm(dataset.train_loader)):
            probabilities = forward_pass(images)
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

def test_model():
    pass
        
train_model(verbose=True)


