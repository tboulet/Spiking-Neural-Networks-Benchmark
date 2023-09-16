from utils import set_seed

import torch
import torch.nn as nn

# Ensure that the total number of parameters is approximately 18,000
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Base model class
from models.model import Model


class TimeSeriesCNN(Model):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

    def build_model(self):
        # Define a Conv1D layer
        self.conv1 = nn.Conv1d(
            in_channels=self.config.n_inputs, out_channels=64, kernel_size=5
        )
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv1d(
            in_channels=32, out_channels=self.config.n_hidden_neurons, kernel_size=3
        )

        # Define a fully connected layer
        self.fc = nn.Linear(
            self.config.n_hidden_neurons,
            self.config.n_hidden_neurons,
            bias=self.config.bias,
        )

        # Add activation function
        self.activation = nn.ReLU()

        # Apply dropout
        self.dropout = nn.Dropout(self.config.dropout_p)

        # Final fully connected layer
        self.fc_final = nn.Linear(self.config.n_hidden_neurons, self.config.n_outputs)

    def forward(self, x):
        # Permute input for Conv1D (batch_size, n_features, n_time) to (batch_size, n_time, n_features)
        x = x.permute(0, 2, 1)

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # Global average pooling to reduce the sequence dimension to 1
        x = nn.functional.avg_pool1d(x, x.size(2))
        x = x.squeeze(2)

        # Apply fully connected layer
        x = self.fc(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Apply final fully connected layer
        x = self.fc_final(x)

        return x

    def init_model(self):
        set_seed(self.config.seed)

    def reset_model(self, train=None):
        pass

    def decrease_sig(self, epoch):
        pass
