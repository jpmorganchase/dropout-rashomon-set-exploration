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

import sys
sys.path.insert(0, '../')
from utils.dropout import change_dropout_rate

def load_cifar10(root, folder):
    # root = './cifar10/data'
    transform = torchvision.transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    train_set = torchvision.datasets.CIFAR10(root=root+folder, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root=root+folder, train=False, download=True)

    X_train, y_train = train_set.data, train_set.targets
    X_test, y_test = test_set.data, test_set.targets
    X_train, X_test = X_train.transpose((0, 3, 1, 2)), X_test.transpose((0, 3, 1, 2))
    ## X.shape = (60000, 3, 32, 32)
    ## y.shape = (60000,)
    return X_train, y_train, X_test, y_test

def load_cifar100(root, folder):
    # root = './cifar100/data'
    transform = torchvision.transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    train_set = torchvision.datasets.CIFAR100(root=root+folder, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR100(root=root+folder, train=False, download=True, transform=transform)

    X_train, y_train = train_set.data, train_set.targets
    X_test, y_test = test_set.data, test_set.targets
    X_train, X_test = X_train.transpose((0, 3, 1, 2)), X_test.transpose((0, 3, 1, 2))
    ## X.shape = (60000, 3, 32, 32)
    ## y.shape = (60000,)
    return X_train, y_train, X_test, y_test

def fetch_model(log, model_name, nclass, pre=True):
    if model_name == 'vgg16':
        model = models.vgg16(pretrained = pre)
        nlayer = len(model.classifier)
        input_lastLayer = model.classifier[nlayer-1].in_features
        model.classifier[nlayer-1] = nn.Linear(input_lastLayer, nclass)
    elif model_name == 'vgg16_bn':
        model = models.vgg16_bn(pretrained = pre)
        nlayer = len(model.classifier)
        input_lastLayer = model.classifier[nlayer-1].in_features
        model.classifier[nlayer-1] = nn.Linear(input_lastLayer, nclass)
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained = pre)
        input_lastLayer = model.fc.in_features
        model.fc = nn.Linear(input_lastLayer, nclass)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained = pre)
        input_lastLayer = model.fc.in_features
        model.fc = nn.Linear(input_lastLayer, nclass)
    elif model_name == 'alexnet':
        model = models.alexnet(pretrained = pre)
        nlayer = len(model.classifier)
        input_lastLayer = model.classifier[nlayer-1].in_features
        model.classifier[nlayer-1] = nn.Linear(input_lastLayer, nclass)
    else:
        log.write('Model {} undefined!\n'.format(model_name))
        log.flush()

    return model 

def build_loader(X, y, shuffle, nbatch):
    X, y = torch.Tensor(X), torch.Tensor(y)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, shuffle=shuffle, batch_size=nbatch)
    return loader

def train(log, args, loader, test_loader, model, criterion, device):
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

        train_acc, train_loss, _ = eval(args, loader, model, criterion, device)
        test_acc, test_loss, _ = eval(args, test_loader, model, criterion, device)

        log.write('Epoch {:3d}/{:3d}: Train loss={:.4f}, Test loss={:.4f}, Train Acc={:.4f}, Test Acc={:.4f}\n'.format(epoch+1, args.nepoch, train_loss, test_loss, train_acc, test_acc))
        log.flush()
        
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