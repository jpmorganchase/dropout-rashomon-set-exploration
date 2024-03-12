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



import sys
sys.path.insert(0, '../')
from utils.loader import load_dataset
from utils.dropout import change_dropout_rate

def get_data_splits(dataset: str, test_size: float, **kwargs):
    X, y = load_dataset(dataset, dropna=True, **kwargs)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=True, stratify=y, random_state=0
    )

    X_train = torch.from_numpy(X_train.values.astype(np.float64))
    X_test = torch.from_numpy(X_test.values.astype(np.float64))
    y_train = torch.from_numpy(y_train.values)
    y_test = torch.from_numpy(y_test.values)

    return X_train, X_test, y_train, y_test

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout()

    def forward(self, x):
        outputs = self.linear(x)
        outputs = self.dropout(outputs)
        return outputs

class sffnn(torch.nn.Module):    
    # Constructor
    def __init__(self, input_dim, hidden_neurons, nlayers, output_dim):
        super(sffnn, self).__init__()
        # hidden layer 
        self.nlayers=nlayers
        self.hidden_neurons=hidden_neurons 
        self.readin = torch.nn.Linear(input_dim, hidden_neurons)
        self.hidden = nn.ModuleList()
        for i in range(self.nlayers - 1):
            self.hidden.append(nn.Linear(hidden_neurons, hidden_neurons))
        self.readout = torch.nn.Linear(hidden_neurons, output_dim) 
        self.act = torch.nn.ReLU()
    
    def forward(self, x):
        outputs = self.readin(x)
        outputs = self.act(outputs)
        for i in range(self.nlayers - 1):
            outputs = self.hidden[i](outputs)
            outputs = self.act(outputs)
        outputs = self.readout(outputs)
        return outputs


def build_loader(X, y, shuffle, nbatch):
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, shuffle=shuffle, batch_size=nbatch)
    return loader

def train(args, loader, model, criterion, device):
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.trainlr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(args.nepoch):
        model.train()  # prep model for training
        for batch_i, (inputs, targets) in enumerate(loader, start=0):
            inputs, targets = inputs.type(torch.float32).to(device), targets.type(torch.LongTensor).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    return model 

def eval(args, loader, model, criterion, device, drp=None):
    if drp != None:
        model = change_dropout_rate(model, drp)
    model.eval()
    losses = 0
    correct = 0
    total = 0
    output = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.type(torch.float32).to(device), targets.type(torch.LongTensor).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            losses += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            output.append(F.softmax(outputs, dim=1))
    losses = losses/(batch_idx+1)
    acc = correct/total
    output = torch.cat(output, dim=0).cpu().detach().numpy()
    return acc, losses, output