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
from .helper import eval

# Adversarial weight perturbation (AWP)
def awp(args, x, loader, model, criterion, loss_tol, k, device):
    optimizer = optim.SGD(model.parameters(), lr=args.awp_lr, momentum=0.9, weight_decay=5e-4)
    model = model.to(device)

    _, loss, _ = eval(args, loader, model, criterion, device)
    cnt = 0
    while loss <= loss_tol:
        _, perturb_loss, _ = eval(args, loader, model, criterion, device)
        perturb_score = model(x)

        model.train()
        optimizer.zero_grad()
        
        x_logit = torch.squeeze(perturb_score)
        objective = -x_logit[k]
        objective.backward()  
        optimizer.step()

        # eval loss after perturbation
        _, loss, _ = eval(args, loader, model, criterion, device)
        cnt += 1
    return perturb_loss, F.softmax(perturb_score, dim=1).cpu().detach().numpy(), cnt


