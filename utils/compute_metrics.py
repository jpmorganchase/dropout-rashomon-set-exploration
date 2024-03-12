# SPDX-License-Identifier: Apache-2.0
# Copyright : J.P. Morgan Chase & Co.
# common packages
import numpy as np
import pickle as pkl
import argparse 
import os
import sys
sys.path.insert(0, '../')
from utils.helper import *
from utils.evaluation import *

from time import localtime, strftime
import time
import copy

# configuration parser
parser = argparse.ArgumentParser(description = "Configuration.")
parser.add_argument('--resultroot', type=str, default='../results/')
## datasets
parser.add_argument('--dataset', type=str, default='adult')
## estimation of predictive multiplicity
parser.add_argument('--model', type=str, default='vgg16', choices=['vgg16', 'vgg16_bn', 'resnet18', 'resnet50', 'alexnet']) 
parser.add_argument('--base_epoch', type=int, default=10) 
parser.add_argument('--method', type=str, default='sampling', choices=['sampling', 'dropout', 'awp']) 
parser.add_argument('--neps', type=int, default=11)
parser.add_argument('--eps_min', type=float, default=0.00)
parser.add_argument('--eps_max', type=float, default=0.04)
## sampling
parser.add_argument('--sampling_nmodel', type=int, default=100)
parser.add_argument('--epoch', type=str, default='')

## dropout
parser.add_argument('--dropoutmethod', type=str, default='bernoulli', choices=['bernoulli', 'gaussian'])
parser.add_argument('--drp_nmodel', type=int, default=100)
parser.add_argument('--drp_max_ratio', type=float, default=0.30) 
## awp
parser.add_argument('--awp_eps', type=str, default='')
args = parser.parse_args()
configuration_dict = vars(args)

# fetch cuda device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# log file and record configs
if args.dataset not in ['cifar10', 'cifar100']: filename_prefix = args.dataset
else: filename_prefix = args.dataset + '-' + args.model

if args.method == 'sampling':
    filename = filename_prefix + '-' + args.method + '-' + str(args.sampling_nmodel) + '-' + args.epoch + '-eval'
elif args.method == 'dropout':
    filename = filename_prefix + '-' + args.dropoutmethod + '-' + args.method + '-' + str(args.drp_nmodel) + '-' + str(args.drp_max_ratio) + '-eval'
elif args.method == 'awp':
    filename = filename_prefix + '-' + args.method + '-' + args.awp_eps + '-eval'

log = open(args.resultroot + filename+'-log.txt', 'w+')
log.write('=== {} ===\n'.format(strftime("%Y-%m-%d-%H.%M.%S", time.localtime())))
log.write('Argument Summary\n')
for key in configuration_dict.keys():
    log.write(' {}: {}\n'.format(key, configuration_dict[key]))

log.write('Device: {}\n'.format(device))
log.flush()

## load base model for comparison
base_data_filename = [i for i in os.listdir(args.resultroot) if filename_prefix+'-base' in i and '.npz' in i]
base_data = np.load(args.resultroot+base_data_filename[0])
base_y_test=base_data['y_test']
base_train_acc=base_data['train_acc']
base_test_acc=base_data['test_acc']
base_train_loss=base_data['train_loss']
base_test_loss=base_data['test_loss']
base_test_scores=base_data['test_scores']
base_train_time=base_data['train_time']

base_pred_y = base_test_scores.argmax(axis=1)

eps_list = np.linspace(0., args.eps_max, args.neps)
# eps_list = np.linspace(0.02, 0.1, args.neps)

if args.method == 'sampling':
    ## 
    epoch_list = np.array([int(i) for i in args.epoch.split(',')])
    nepoch = len(epoch_list)

    sampling_test_loss = []
    sampling_test_scores = []

    ## load all loss and scores
    for i, epoch in enumerate(epoch_list):
        sampling_data = np.load(args.resultroot+filename_prefix+'-sampling-' + str(args.sampling_nmodel) + '-' + str(epoch) +'.npz')

        sampling_test_loss.append(sampling_data['test_losses_acc'][:, 0])
        sampling_test_scores.append(sampling_data['test_scores'])

    all_sampling_test_loss = sampling_test_loss[0]
    all_sampling_test_scores = sampling_test_scores[0]
    for i in range(1, len(sampling_test_loss)):
        all_sampling_test_loss = np.concatenate((all_sampling_test_loss, sampling_test_loss[i]), axis=0)
        all_sampling_test_scores = np.concatenate((all_sampling_test_scores, sampling_test_scores[i]), axis=0)

    ## compute predictive multiplicity metrics
    ntest = all_sampling_test_scores.shape[1]

    vpr = np.zeros((args.neps, ntest))
    score_var = np.zeros((args.neps, ntest))
    rc = np.zeros((args.neps, ntest))

    amb = np.zeros((args.neps, ))
    disc = np.zeros((args.neps, ))
    disa_hat = np.zeros((args.neps, ntest))

    for i, eps in enumerate(eps_list):
        log.write('{:3d}/{:3d} epsilon = {:.2f}\n'.format(i+1, len(eps_list), eps))
        log.flush()

        loss_tol = base_test_loss + eps
        scores = get_Rashomon_models_sampling(all_sampling_test_loss, loss_tol, all_sampling_test_scores)
        if scores.size == 0: continue 
        score_y = score_of_y_multi_model(scores, base_y_test)
        # score_y = scores[:, :, 1] ## only for binary classification, used in some papers
        # base_pred_y = base_test_scores.argmax(axis=1)

        ## score-based
        vpr[i, :] = viable_prediction_range(score_y)
        score_var[i, :] = score_variance(score_y)
        rc[i, :] = rashomon_capacity(scores)
        
        ## decision-based
        decisions = scores.argmax(axis=2)
        amb[i] = ambiguity(decisions, base_pred_y)
        disc[i] = discrepancy(decisions, base_pred_y)
        disa_hat[i, :] = disagreement_hat(decisions)

    savename = filename + '.npz'

    np.savez_compressed(args.resultroot+savename,
                        all_sampling_test_loss=all_sampling_test_loss,
                        all_sampling_test_scores=all_sampling_test_scores,
                        eps_list=eps_list,
                        vpr=vpr,
                        score_var=score_var,
                        rc=rc,
                        amb=amb,
                        disc=disc,
                        disa_hat=disa_hat
                        )


elif args.method == 'dropout':
    drp_data_filename = [i for i in os.listdir(args.resultroot) if filename_prefix + '-' + args.dropoutmethod + '-' + args.method in i and str(args.drp_nmodel) + '-' + str(args.drp_max_ratio)+'.npz' in i]
    drp_data = np.load(args.resultroot+drp_data_filename[0])
    drp_y_test=drp_data['y_test']
    drp_train_acc=drp_data['train_acc']
    drp_train_loss=drp_data['train_loss']
    drp_test_acc=drp_data['test_acc']
    drp_test_loss=drp_data['test_loss']
    drp_drp_test_acc=drp_data['drp_test_acc']
    drp_drp_test_loss=drp_data['drp_test_loss']
    drp_all_test_scores=drp_data['all_test_scores']
    drp_inference_time=drp_data['inference_time']
    drp_drp_list=drp_data['drp_list']

    ndrp = drp_drp_test_loss.shape[0]

    drp_test_loss = []
    drp_test_scores = []

    ## load all loss and scores
    for i in range(ndrp):
        drp_test_loss.append(drp_drp_test_loss[i, :])
        drp_test_scores.append(drp_all_test_scores[i, :, :, :])

    all_drp_test_loss = drp_test_loss[0]
    all_drp_test_scores = drp_test_scores[0]
    for i in range(1, len(drp_test_loss)):
        all_drp_test_loss = np.concatenate((all_drp_test_loss, drp_test_loss[i]), axis=0)
        all_drp_test_scores = np.concatenate((all_drp_test_scores, drp_test_scores[i]), axis=0)

    ## compute predictive multiplicity metrics
    ntest = all_drp_test_scores.shape[1]

    vpr = np.zeros((args.neps, ntest))
    score_var = np.zeros((args.neps, ntest))
    rc = np.zeros((args.neps, ntest))

    amb = np.zeros((args.neps, ))
    disc = np.zeros((args.neps, ))
    disa_hat = np.zeros((args.neps, ntest))

    for i, eps in enumerate(eps_list):
        log.write('{:3d}/{:3d} epsilon = {:.2f}\n'.format(i+1, len(eps_list), eps))
        log.flush()

        loss_tol = base_test_loss + eps
        scores = get_Rashomon_models_sampling(all_drp_test_loss, loss_tol, all_drp_test_scores)
        if scores.size == 0: continue 

        score_y = score_of_y_multi_model(scores, base_y_test)
        
        # score_y = scores[:, :, 1] ## only for binary classification, used in some papers
        # base_pred_y = base_test_scores.argmax(axis=1)

        ## score-based
        vpr[i, :] = viable_prediction_range(score_y)
        score_var[i, :] = score_variance(score_y)
        rc[i, :] = rashomon_capacity(scores)
        
        ## decision-based
        decisions = scores.argmax(axis=2)
        amb[i] = ambiguity(decisions, base_pred_y)
        disc[i] = discrepancy(decisions, base_pred_y)
        disa_hat[i, :] = disagreement_hat(decisions)

    savename = filename + '.npz'

    np.savez_compressed(args.resultroot+savename,
                        all_drp_test_loss=all_drp_test_loss,
                        all_drp_test_scores=all_drp_test_scores,
                        eps_list=eps_list,
                        vpr=vpr,
                        score_var=score_var,
                        rc=rc,
                        amb=amb,
                        disc=disc,
                        disa_hat=disa_hat
                        )

elif args.method == 'awp':
    awp_data = np.load(args.resultroot + filename_prefix + '-' + args.method + '-' + args.awp_eps + '.npz')
    y_test=awp_data['y_test']
    eps_list=awp_data['eps_list']
    base_loss=awp_data['base_loss']
    perturb_scores=awp_data['perturb_scores'] # neps * ntest * nclass * nclass 
    perturb_losses=awp_data['perturb_losses']
    perturb_time=awp_data['perturb_time']

    neps = len(eps_list)
    ntest = len(y_test)

    score_y = score_of_y_multi_model_awp(perturb_scores, base_y_test)
    decisions = perturb_scores.argmax(axis=3)

    vpr = np.zeros((neps, ntest))
    score_var = np.zeros((neps, ntest))
    rc = np.zeros((neps, ntest))

    amb = np.zeros((neps, ))
    disc = np.zeros((neps, ))
    disa_hat = np.zeros((neps, ntest))

    for i, eps in enumerate(eps_list):
        log.write('{:3d}/{:3d} epsilon = {:.2f}\n'.format(i+1, len(eps_list), eps))
        log.flush()

        vpr[i, :] = viable_prediction_range_awp(score_y[i, :, :])
        score_var[i, :] = score_variance_awp(score_y[i, :, :])
        rc[i, :] = rashomon_capacity_awp(perturb_scores[i, :, :, :])

        amb[i] = ambiguity_awp(decisions[i, :, :], base_pred_y) 
        disc[i] = discrepancy_awp(decisions[i, :, :], base_pred_y)
        disa_hat[i] = disagreement_hat_awp(decisions[i, :, :])

    savename = filename + '.npz'

    np.savez_compressed(args.resultroot+savename,
                        perturb_scores=perturb_scores,
                        perturb_losses=perturb_losses,
                        eps_list=eps_list,
                        vpr=vpr,
                        score_var=score_var,
                        rc=rc,
                        amb=amb,
                        disc=disc,
                        disa_hat=disa_hat
                        )

else:
    print('Method {} not supported yet!!!\n'.format(args.method))    

print('Finished!!!')
log.write('Finished!!!\n')
log.close()