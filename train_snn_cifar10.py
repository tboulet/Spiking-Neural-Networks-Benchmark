from datasets import cifar10_dataloaders
from config_snn_cifar10 import Config
from snn_delays import SnnDelays
import torch
from snn import SNN
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n=====> Device = {device} \n\n")

config = Config()

if config.model_type == 'snn':
    model = SNN(config).to(device)
else:
    model = SnnDelays(config).to(device)

if config.model_type == 'snn_delays_lr0':
    model.round_pos()


if config.model_type == 'snn_delays_lr0':
    model.round_pos()


print(f"===> Dataset    = {config.dataset}")
print(f"===> Model type = {config.model_type}")
print(f"===> Model size = {utils.count_parameters(model)}\n\n")


train_loader, valid_loader, test_loader = cifar10_dataloaders(config)

if __name__ == '__main__':
    model.train_model(train_loader, valid_loader, test_loader, device)