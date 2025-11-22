
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


__all__ = ['test_ds', 'train_ds', 'validation_ds', 'test_loader', 'train_loader', 'validation_loader']

dataset_dir = '../../res'

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081))]
    )

initial_train_ds = datasets.MNIST(
    root = dataset_dir,
    train = True,
    download = True,
    transform = transform
    )

test_ds = datasets.MNIST(
    root = dataset_dir,
    train = False,
    download = False,
    transform = transform
    )

train_size = 50_000
validation_size = 10_000
train_ds, validation_ds = random_split(initial_train_ds, [train_size, validation_size]) 

test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
validation_loader = DataLoader(validation_ds, batch_size=64, shuffle=False)


def load_model():
    pass

def save_model():
    pass
