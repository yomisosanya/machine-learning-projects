from torchvision import datasets 
from torchvision import transforms

dataset_dir = '../../res/mnist'

transform = transforms.Compose(
    [transforms.To_Tensor(),
    transforme.Normalize((0.1307,), (0.3081))]
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