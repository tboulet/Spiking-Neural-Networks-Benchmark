import os

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

SEED = 0
np.random.seed(SEED)


class BinaryTimeSeries(Dataset):
    def __init__(
        self, x, y, split_name, transform=None, target_transform=None, download=True
    ):
        self.split_name = split_name
        self.transform = transform
        self.target_transform = target_transform

        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]

        if self.transform is not None:
            x = self.transform(x).squeeze().t()

        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y, torch.zeros(1)


labels = ["Jogging", "Walking", "Upstairs", "Downstairs", "Sitting", "Standing"]


def target_transform(updown):
    return torch.tensor(labels.index(updown))


def HAR_dataloaders(config):

    current_dir = os.path.dirname(os.path.abspath(__file__))

    x_train = np.load(os.path.join(current_dir, "raw/x_train_WISDM.npy"))
    y_train = pd.read_csv(os.path.join(current_dir, "raw/y_train_WISDM.csv")).values

    x_test = np.load(os.path.join(current_dir, "raw/x_test_WISDM.npy"))
    y_test = pd.read_csv(os.path.join(current_dir, "raw/y_test_WISDM.csv")).values

    # Combine the train and test sets
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    full_length = len(y)
    size = 0.4

    x = x[: int(full_length * size)]
    y = y[: int(full_length * size)]

    length = len(y)

    # Define the ratios for train, test, and validation sets
    train_ratio = 0.7
    test_ratio = 0.15
    val_ratio = 0.15

    # Calculate the number of indices for each set
    num_train = int(length * train_ratio)
    num_test = int(length * test_ratio)
    num_val = length - num_train - num_test

    print(num_train, num_test, num_val)

    # Generate a random permutation of indices
    indices = np.random.permutation(length)

    # Split the indices into train, test, and validation sets
    train_indices = indices[:num_train]
    test_indices = indices[num_train : num_train + num_test]
    val_indices = indices[num_train + num_test :]

    x_train, y_train = x[train_indices], y[train_indices]
    x_test, y_test = x[test_indices], y[test_indices]
    x_val, y_val = x[val_indices], y[val_indices]

    train_dataset = BinaryTimeSeries(
        x_train, y_train, "training", target_transform=target_transform
    )
    valid_dataset = BinaryTimeSeries(
        x_val, y_val, "validation", target_transform=target_transform
    )
    test_dataset = BinaryTimeSeries(
        x_test, y_test, "testing", target_transform=target_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    return train_loader, valid_loader, test_loader
