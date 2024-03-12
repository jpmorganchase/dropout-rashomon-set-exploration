# SPDX-License-Identifier: Apache-2.0
# Copyright : J.P. Morgan Chase & Co.
import pandas as pd
import numpy as np
import scipy as sp

##
from time import localtime, strftime
import time

## pytorch
import torch
from torch import nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import models

from sklearn.model_selection import train_test_split

## dropout
def add_dropout(model: nn.Module, method: str) -> nn.Module:
    if method == 'bernoulli':
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.register_forward_hook(BernoulliDropout)
                layer.p = 0.0
    elif method == 'gaussian':
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.register_forward_hook(GaussianDropout)
                layer.p = 0.0
    else:
        print('dropout method {} does not exist'.format(method))
        return 
    return model 

def BernoulliDropout(layer, input, output): ## Bernoulli dropout
    output = F.dropout(output, p=layer.p, training=True)
    return output 

def GaussianDropout(layer, input, output): ## Gaussian dropout
    output = output * (torch.randn_like(output) * layer.p + 1)
    return output 

def change_dropout_rate(model: nn.Module, p: float) -> nn.Module:
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.p = p
    return model 


