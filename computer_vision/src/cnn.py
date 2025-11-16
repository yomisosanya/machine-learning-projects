import torch
import torch.nn as nn
from torchmetrics import Accuracy, Precision, Recall
import dataset
import sys
from tqdm import tqdm
from typing import TextIO

c0 = 1
c1 = 64
c2 = 256
fc0 = 516
cls = 10  # number of classes

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.filter = nn.Sequential(
            # grayscale immage, out-channel = 64, simple feature like edgeslarge 
            # large kernel size to focus more on general shape
            nn.Conv2d(in_channels=c0, out_channels=c1, kernel_size=4, padding=2), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # out-channel = 256, more complex feature and details
            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(c2 * 16, fc0),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(fc0, fc0),
            nn.RelU(),
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


def run(epochs: int, verbose: bool, out: TextIO = sys.stdout) :

    kwargs = {'task': 'multiclass', 'num_classes': cls}
    accuracy = Accuracy(**kwargs)
    precision = Precision(average='macro', **kwargs)
    recall = Recall(average='macro', **kwargs)
    if verbose:
        pass
    
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}', file=out)

        for (images, labels) in dataset.train:
            probabilities = model(images, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            accuracy.update(predictions, labels)
            precision.update(predictions, labels)
            recall.update(predictions, labels)
        print(f'Epoch accuracy: {accuracy.compute()}')
        print(f'Epoch precision: {precision.compute()}')
        print(f'Epoch recall: {recall.compute()}')
        accuracy.reset()
        precision.reset()
        recall.reset()
        



