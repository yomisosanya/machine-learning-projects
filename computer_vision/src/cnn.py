from torch import nn
from torchvision import datasets 
from torchvision import transforms


dataset_dir = '../../res'

transform = transforms.Compose(
    [transforms.To_Tensor(),
    transforms.Normalize((0.1307,), (0.3081))]
)

dataset_train = datasets(
    root = dataset_dir
    train = True
    download = False
    transform = transform
)

dataset_test = datasets(
    root = dataset_dir
    train = False
    download = False
    transform = transform
)


class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential()
        self.layer2 = nn.Sequential()
        self.fully_conn = nn.Linear()

    def forward(self, input):
        output = self.layer1(input)
        output = self.layer2(output)
        output = output.reshape()
        output = self.fully_conn(output)
        return output