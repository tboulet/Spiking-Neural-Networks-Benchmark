from utils import set_seed

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms

class Flatten(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """ """
        return tensor.flatten()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class Repeat(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """ """
        return tensor.repeat(self.n, 1)


def cifar10_dataloaders(config, shuffle=True, valid_size=0.2, num_workers=4):
    set_seed(config.seed)
    batch_size = config.batch_size

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Converts images to tensors
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ),  # Normalize image pixel values
            Flatten(),
            Repeat(config.max_delay),
        ]
    )

    # Download CIFAR-10 dataset if not already downloaded
    train_dataset = torchvision.datasets.CIFAR10(
        root=config.datasets_path, train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=config.datasets_path, train=False, transform=transform, download=True
    )

    # Split the training dataset into training and validation sets
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(valid_size * num_train)
    if shuffle:
        torch.manual_seed(42)  # For reproducibility
        indices = torch.randperm(num_train)

    # Split the training dataset into training and validation sets
    train_indices, valid_indices = indices[split:], indices[:split]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)

    # Create data loaders for training, validation, and testing
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Print the dataset sizes and shapes
    print(f"Train dataset size: {len(train_loader.sampler)}")
    print(f"Validation dataset size: {len(valid_loader.sampler)}")
    print(f"Test dataset size: {len(test_loader.sampler)}")
    print(f"Image shape: {train_dataset[0][0].shape}")
    print(f"Number of classes: {len(train_dataset.classes)}")

    return train_loader, valid_loader, test_loader


def cifar10_repeat_dataloaders(config, shuffle=True, valid_size=0.2, num_workers=4):
    set_seed(config.seed)
    batch_size = config.batch_size

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Converts images to tensors
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ),  # Normalize image pixel values
            Flatten(),
            Repeat(config.max_delay),
        ]
    )

    # Download CIFAR-10 dataset if not already downloaded
    train_dataset = torchvision.datasets.CIFAR10(
        root=config.datasets_path, train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=config.datasets_path, train=False, transform=transform, download=True
    )

    # Split the training dataset into training and validation sets
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(valid_size * num_train)
    if shuffle:
        torch.manual_seed(42)  # For reproducibility
        indices = torch.randperm(num_train)

    # Split the training dataset into training and validation sets
    train_indices, valid_indices = indices[split:], indices[:split]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)

    # Create data loaders for training, validation, and testing
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Print the dataset sizes and shapes
    print(f"Train dataset size: {len(train_loader.sampler)}")
    print(f"Validation dataset size: {len(valid_loader.sampler)}")
    print(f"Test dataset size: {len(test_loader.sampler)}")
    print(f"Image shape: {train_dataset[0][0].shape}")
    print(f"Number of classes: {len(train_dataset.classes)}")

    return train_loader, valid_loader, test_loader
