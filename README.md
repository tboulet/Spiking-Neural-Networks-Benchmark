#  Learning Delays in Spiking Neural Networks using Dilated Convolutions with Learnable Spacings.

This project is a ripoff from this github: https://github.com/Thvnvtos/SNN-delays

## Dependencies
### Python
Please use Python 3.9 and above ```python>=3.9```

To use the repository, we need to install the requirements and the spikingjelly package.
### Requirements
Install all the dependencies in a new python environment
'''
python -m venv snnvenv
pip install -e .
'''
### SpikingJelly
Install SpikingJelly using:
```
git clone https://github.com/fangwei123456/spikingjelly.git
cd spikingjelly
pip install -e .
```



## Introduction
This repository is meant to test the snn with and without delays on cifar10 or a times series datasets like Human Activity Recognition dataset and some other and compare the performance to traditional artificial neural network. Other dataset can be added for further experiments.

### Usage
To train a model on a dataset, simply use the command python ./snn_benchmark/train.py --config_name=CONFIG_NAME where config_name is the name of the config that you want to use. All of them are defined inside of the train.py file. To add another one. You have to add your config into the configs file, add it into the __init__.py and inside of the train.py to be able to call it.

The dataset will be downloaded automatically if necessary as mentioned inside of the config.

### Logs

The loss and accuracy for the training and validation at every epoch will be printed to ```stdout``` and the best model will be saved to the current directory.
If the ```use_wandb``` parameter is set to ```True```, a more detailed log will be available at the wandb project specified in the configuration.