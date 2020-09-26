import torchvision


def load_mnist(root="data/mnist", train=True, download=True):
    """"""
    dataset = torchvision.datasets.MNIST(
        root,
        train=train,
        transform=torchvision.transforms.ToTensor(),
        target_transform=None,
        download=download)

    return dataset
