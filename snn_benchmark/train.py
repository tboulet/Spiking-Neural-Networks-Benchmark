import fire

import torch
import utils

from data.base_datasets import SHD_dataloaders, SSC_dataloaders, GSC_dataloaders
from data.cifar_dataset import cifar10_repeat_dataloaders, cifar10_dataloaders
from data.iris_dataset import IRIS_dataloaders
from data.har_dataset import HAR_dataloaders

from models.ann import ANN
from models.snn import SNN
from models.snn_delays import SnnDelays
from models.conv import TimeSeriesCNN

from configs import (
    Config,
    ConfigGSC,
    ConfigSSC,
    ConfigSHD,
    ConfigIRIS,
    ConfigSNNIRIS,
    ConfigANNCIFAR10,
    ConfigSNNCIFAR10,
    ConfigSNNCIFAR10LINE,
    ConfigSNNREPEATCIFAR10,
    ConfigHAR
)

config_mapping = {
    "GSC": (ConfigGSC, GSC_dataloaders, None),
    "SSC": (ConfigSSC, SSC_dataloaders, None),
    "SHD": (ConfigSHD, SHD_dataloaders, None),
    "IRIS": (ConfigIRIS, IRIS_dataloaders, ANN),
    "SNN_IRIS": (ConfigSNNIRIS, IRIS_dataloaders, None),
    "ANN_CIFAR10": (ConfigANNCIFAR10, cifar10_dataloaders, ANN),
    "SNN_CIFAR10": (ConfigSNNCIFAR10, cifar10_dataloaders, None),
    "SNN_CIFAR10_LINE": (ConfigSNNCIFAR10LINE, cifar10_dataloaders, None),
    "SNN_REPEAT_CIFAR10": (ConfigSNNREPEATCIFAR10, cifar10_repeat_dataloaders, None),
    "SNN_HAR": (ConfigHAR, HAR_dataloaders, None),
    # "CONV1D_HAR": (Config, HAR_dataloaders, TimeSeriesCNN),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n=====> Device = {device} \n\n")


def launch(config_name='CONV1D_HAR'):
    config, dataloaders, model = config_mapping.get(config_name, (Config, HAR_dataloaders, None))
    train_loader, valid_loader, test_loader = dataloaders(config)

    if model is None:
        model = SNN(config) if config.model_type == "snn" else SnnDelays(config)
        model.to(device)

        if config.model_type == "snn_delays_lr0":
            model.round_pos()
    else:
        model = model(config)
        model.to(device)

    print(f"===> Dataset    = {config.dataset}")
    print(f"===> Model type = {config.model_type}")
    print(f"===> Model size = {utils.count_parameters(model)}\n\n")

    model.train_model(train_loader, valid_loader, test_loader, device)

if __name__ == "__main__":
    fire.Fire(launch)
