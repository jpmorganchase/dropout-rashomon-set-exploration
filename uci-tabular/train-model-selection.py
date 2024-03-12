# SPDX-License-Identifier: Apache-2.0
# Copyright : J.P. Morgan Chase & Co.
# common packages
import numpy as np
import pickle as pkl
import argparse 
import sys
 
sys.path.insert(0, '../')
from utils.helper import *

from time import localtime, strftime
import time
import copy

# configuration parser
parser = argparse.ArgumentParser(description = "Configuration.")
parser.add_argument('--resultroot', type=str, default='../results/')
## datasets
parser.add_argument('--dataset', type=str, default='adult')
parser.add_argument('--test_size', type=float, default=0.25) 
## nn architecture
parser.add_argument('--nneuron', type=int, default=1000) 
parser.add_argument('--nhiddenlayer', type=int, default=1) 
## training and inference 
parser.add_argument('--nepoch', type=int, default=100) 
parser.add_argument('--train_batch_size', type=int, default=100) 
parser.add_argument('--test_batch_size', type=int, default=1000) 
parser.add_argument('--trainlr', type=float, default=1e-3) 
## sampling
parser.add_argument('--nretraining', type=int, default=10)
## dropout
parser.add_argument('--dropoutmethod', type=str, default='bernoulli', choices=['bernoulli', 'gaussian'])
parser.add_argument('--drp_nmodel', type=int, default=100)
parser.add_argument('--ndrp', type=int, default=21)
parser.add_argument('--drp_max_ratio', type=float, default=0.30) 
args = parser.parse_args()
configuration_dict = vars(args)

# fetch cuda device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

filename = args.dataset + '-' + str(args.nretraining) + '-' + str(args.nepoch) + '-' + args.dropoutmethod + '-model-selection'


log = open(args.resultroot + filename+'-log.txt', 'w+')
log.write('=== {} ===\n'.format(strftime("%Y-%m-%d-%H.%M.%S", time.localtime())))
log.write('Argument Summary\n')
for key in configuration_dict.keys():
    log.write(' {}: {}\n'.format(key, configuration_dict[key]))

log.write('Device: {}\n'.format(device))
log.flush()

## loading dataset
X_train, X_test, y_train, y_test = get_data_splits(args.dataset, args.test_size)
log.write('Train # = {}\n'.format(X_train.shape[0]))
log.write('Test # = {}\n'.format(X_test.shape[0]))
log.write('feature # = {}\n'.format(X_train.shape[1]))
log.write('class # = {}\n'.format(len(np.unique(y_train))))
log.flush()

## build dataloaders
trainloader = build_loader(X_train, y_train, shuffle=True, nbatch=args.train_batch_size)
testloader = build_loader(X_test, y_test, shuffle=False, nbatch=args.test_batch_size)

## training
ntest = len(y_test)
ntrain = len(y_train)
input_dim = X_train.shape[1]
output_dim = len(np.unique(y_train))
criterion = torch.nn.CrossEntropyLoss()

drp_list = np.linspace(0.00, args.drp_max_ratio, args.ndrp)


train_loss = np.zeros((args.nretraining,))
train_acc = np.zeros((args.nretraining,))
test_loss = np.zeros((args.nretraining,))
test_acc = np.zeros((args.nretraining,))
test_scores = np.zeros((args.nretraining, ntest, output_dim))


drp_test_acc = np.zeros((args.nretraining, len(drp_list), args.drp_nmodel, ))
drp_test_loss = np.zeros((args.nretraining, len(drp_list), args.drp_nmodel, ))
all_test_scores = np.zeros((args.nretraining, len(drp_list), args.drp_nmodel, ntest, output_dim))

for i in range(args.nretraining):
    model = sffnn(input_dim, args.nneuron, args.nhiddenlayer, output_dim)
    model = train(args, trainloader, model, criterion, device)

    train_acc[i], train_loss[i], _ = eval(args, trainloader, model, criterion, device)
    test_acc[i], test_loss[i], test_scores[i, :, :] = eval(args, testloader, model, criterion, device)

    log.write('Model {:3d}/{:3d}: Train loss={:.4f}, Test loss={:.4f}, Train Acc={:.4f}, Test Acc={:.4f}\n'.format(i+1, args.nretraining, train_loss[i], test_loss[i], train_acc[i], test_acc[i]))
    log.flush()

    model = add_dropout(model, method=args.dropoutmethod)
    for j, drp in enumerate(drp_list):
        for k in range(args.drp_nmodel):
            drp_test_acc[i, j, k], drp_test_loss[i, j, k], all_test_scores[i, j, k, :, :] = eval(args, testloader, model, criterion, device, drp=drp)
        
        log.write(' Dropout = {:.4f}, Test Loss = {:.4f}, Test Acc = {:.4f}\n'.format(drp, drp_test_loss[i, j, :].mean(), drp_test_acc[i, j, :].mean()))
        log.flush()


savename = filename + '.npz'

np.savez_compressed(args.resultroot+savename,
                    y_test=y_test,
                    train_acc=train_acc,
                    train_loss=train_loss,
                    test_acc=test_acc,
                    test_loss=test_loss,
                    test_scores=test_scores,
                    drp_test_acc=drp_test_acc,
                    drp_test_loss=drp_test_loss,
                    all_test_scores=all_test_scores,
                    drp_list=drp_list)


print('Finished!!!')
log.write('Finished!!!\n')
log.close()