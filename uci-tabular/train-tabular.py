# SPDX-License-Identifier: Apache-2.0
# Copyright : J.P. Morgan Chase & Co.
import numpy as np
import pickle as pkl
import argparse 
import sys
import os

from time import localtime, strftime
import time
import copy
 
sys.path.insert(0, '../')
from utils.helper import *
from utils.dropout import add_dropout
from utils.awp import awp

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
## estimation of predictive multiplicity
parser.add_argument('--method', type=str, default='sampling', choices=['base', 'sampling', 'dropout', 'awp']) 
## sampling
parser.add_argument('--sampling_nmodel', type=int, default=100)
## dropout
parser.add_argument('--dropoutmethod', type=str, default='bernoulli', choices=['bernoulli', 'gaussian'])
parser.add_argument('--drp_nmodel', type=int, default=100)
parser.add_argument('--drp_max_ratio', type=float, default=0.30) 
## awp
parser.add_argument('--awp_eps', type=str, default='') 
parser.add_argument('--awp_lr', type=float, default=1e-3)
args = parser.parse_args()
configuration_dict = vars(args)

# fetch cuda device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# log file and record configs
if args.method == 'base':
    filename = args.dataset + '-' + args.method + '-' + str(args.nepoch) + '-' + str(args.nhiddenlayer) + '-' + str(args.nneuron)
elif args.method == 'sampling':
    filename = args.dataset + '-' + args.method + '-' + str(args.sampling_nmodel) + '-' + str(args.nepoch)
elif args.method == 'dropout':
    filename = args.dataset + '-' + args.dropoutmethod + '-' + args.method + '-' + str(args.nepoch) + '-' + str(args.nhiddenlayer) + '-' + str(args.nneuron) + '-' + str(args.drp_nmodel) + '-' + str(args.drp_max_ratio)
elif args.method == 'awp':
    filename = args.dataset + '-' + args.method + '-' + args.awp_eps

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

if args.method == 'base':
    model = sffnn(input_dim, args.nneuron, args.nhiddenlayer, output_dim)
    train_start_time = time.localtime()
    model = train(args, trainloader, model, criterion, device)
    train_time = time.mktime(time.localtime())-time.mktime(train_start_time)

    train_acc, train_loss, train_scores = eval(args, trainloader, model, criterion, device, drp=0.0)
    test_acc, test_loss, test_scores = eval(args, testloader, model, criterion, device, drp=0.0)

    log.write('({:.2f} secs): Train loss={:.4f}, Test loss={:.4f}, Train Acc={:.4f}, Test Acc={:.4f}\n'.format(train_time, train_loss, test_loss, train_acc, test_acc))
    log.flush()

    torch.save(model.state_dict(), args.resultroot + filename + '.pt')

    savename = filename + '.npz'

    np.savez_compressed(args.resultroot+savename,
                        y_test=y_test,
                        train_acc=train_acc,
                        test_acc=test_acc,
                        train_loss=train_loss,
                        test_loss=test_loss,
                        test_scores=test_scores,
                        train_time=train_time)


elif args.method == 'sampling':
    train_losses_acc = np.zeros((args.sampling_nmodel, 2))
    test_losses_acc = np.zeros((args.sampling_nmodel, 2))
    test_scores = np.zeros((args.sampling_nmodel, ntest, output_dim))
    train_time = np.zeros((args.sampling_nmodel,))

    for i in range(args.sampling_nmodel):
        model = sffnn(input_dim, args.nneuron, args.nhiddenlayer, output_dim)
        train_start_time = time.localtime()
        model = train(args, trainloader, model, criterion, device)
        train_end_time = time.localtime()
        train_time[i] = time.mktime(train_end_time)-time.mktime(train_start_time)

        train_losses_acc[i, 1], train_losses_acc[i, 0], _ = eval(args, trainloader, model, criterion, device)
        test_losses_acc[i, 1], test_losses_acc[i, 0], test_scores[i, :, :] = eval(args, testloader, model, criterion, device)

        log.write('Model {:3d}/{:3d} ({:.2f} secs): Train loss={:.4f}, Test loss={:.4f}, Train Acc={:.4f}, Test Acc={:.4f}\n'.format(i+1, args.sampling_nmodel, train_time[i], train_losses_acc[i, 0], test_losses_acc[i, 0], train_losses_acc[i, 1], test_losses_acc[i, 1]))
        log.flush()

    savename = filename + '.npz'

    np.savez_compressed(args.resultroot+savename,
                        y_test=y_test,
                        train_losses_acc=train_losses_acc,
                        test_losses_acc=test_losses_acc,
                        test_scores=test_scores,
                        train_time=train_time)
    

elif args.method == 'dropout':
    pretrain_name = [i for i in os.listdir(args.resultroot) if args.dataset+'-base' in i and '.pt' in i]
    ## load pre-trained model
    model = sffnn(input_dim, args.nneuron, args.nhiddenlayer, output_dim)
    model.load_state_dict(torch.load(args.resultroot+pretrain_name[0]))
    model = add_dropout(model, method=args.dropoutmethod)
    model = model.to(device)

    ## evaluation
    train_acc, train_loss, _ = eval(args, trainloader, model, criterion, device)
    test_acc, test_loss, test_scores_base = eval(args, testloader, model, criterion, device)
    log.write('Loading pre-train model: {}\n'.format(args.resultroot+pretrain_name[0]))
    log.write('Train Acc = {:.4f}, Loss  = {:.4f}\n'.format(train_acc, train_loss))
    log.write('Test Acc = {:.4f}, Loss  = {:.4f}\n'.format(test_acc, test_loss))
    log.flush()

    drp_list = np.linspace(0.00, args.drp_max_ratio, 21)
    drp_test_acc = np.zeros((len(drp_list), args.drp_nmodel, ))
    drp_test_loss = np.zeros((len(drp_list), args.drp_nmodel, ))

    all_test_scores = np.zeros((len(drp_list), args.drp_nmodel, ntest, output_dim))
    inference_time = np.zeros((len(drp_list), args.drp_nmodel))

    for i, drp in enumerate(drp_list):
        for j in range(args.drp_nmodel):
            inference_start_time = time.localtime()
            drp_test_acc[i, j], drp_test_loss[i, j], all_test_scores[i, j, :, :] = eval(args, testloader, model, criterion, device, drp=drp)
            inference_time[i, j] = time.mktime(time.localtime())-time.mktime(inference_start_time)
        
        log.write('Dropout = {:.4f}, Time = {:.4f} secs, Test Loss = {:.4f}, Test Acc = {:.4f}\n'.format(drp, inference_time[i, :].mean(), drp_test_loss[i, :].mean(), drp_test_acc[i, :].mean()))
        log.flush()

    savename = filename + '.npz'

    np.savez_compressed(args.resultroot+savename,
                        y_test=y_test,
                        train_acc=train_acc,
                        train_loss=train_loss,
                        test_acc=test_acc,
                        test_loss=test_loss,
                        drp_test_acc=drp_test_acc,
                        drp_test_loss=drp_test_loss,
                        all_test_scores=all_test_scores,
                        inference_time=inference_time,
                        drp_list=drp_list)
    
elif args.method == 'awp':
    pretrain_name = [i for i in os.listdir(args.resultroot) if args.dataset+'-base' in i and '.pt' in i]
    eps_list = np.array([float(i) for i in args.awp_eps.split(',')])
    neps = len(eps_list)

    base_model = sffnn(input_dim, args.nneuron, args.nhiddenlayer, output_dim)
    base_model.load_state_dict(torch.load(args.resultroot+pretrain_name[0]))
    base_model = base_model.to(device)
    _, base_loss, _ = eval(args, testloader, base_model, criterion, device)

    perturb_scores = np.zeros((neps, ntest, output_dim, output_dim))
    perturb_losses = np.zeros((neps, ntest, output_dim))
    perturb_time = np.zeros((neps, ntest))

    for i, eps in enumerate(eps_list):
        loss_tol = base_loss + eps
        for j in range(ntest):
            x_target = X_test[j, :].reshape((1, input_dim))
            x_target = torch.Tensor(x_target.type(torch.float32)).to(device)

            perturb_start_time = time.localtime()
            perturb_cnt = 0
            for k in range(output_dim): # iterate over classes
                model = sffnn(input_dim, args.nneuron, args.nhiddenlayer, output_dim)
                model.load_state_dict(copy.deepcopy(base_model.state_dict()))
                ## awp for a single sample for a single class
                perturb_losses[i, j, k], perturb_scores[i, j, k, :], cnt = awp(args, x_target, testloader, model, criterion, loss_tol, k, device)
                perturb_cnt += cnt

            perturb_time[i, j] = time.mktime(time.localtime())-time.mktime(perturb_start_time)
            log.write('{:3d}/{:3d} epsilon = {:.4f} Sample {:5d}/{:5d} perturb {:3d} ({:.2f} secs)\n'.format(i+1, neps, eps, j+1, ntest, perturb_cnt, perturb_time[i, j]))
            log.flush()

    savename = filename + '.npz'

    np.savez_compressed(args.resultroot+savename,
                        y_test=y_test,
                        eps_list=eps_list,
                        base_loss=base_loss,
                        perturb_scores=perturb_scores,
                        perturb_losses=perturb_losses,
                        perturb_time=perturb_time)


else:
    print('Method {} not supported yet!!!\n'.format(args.method))
    
print('Finished!!!')
log.write('Finished!!!\n')
log.close()