from torchvision import datasets, transforms

dataset_dir = '../../res'

transform = transforms.Compose(
    [transforms.To_Tensor(),
    transforms.Normalize((0.1307,), (0.3081))]
)

train = datasets(
    root = dataset_dir
    train = True
    download = False
    transform = transform
)

test = datasets(
    root = dataset_dir
    train = False
    download = False
    transform = transform
)
