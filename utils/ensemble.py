# SPDX-License-Identifier: Apache-2.0
# Copyright : J.P. Morgan Chase & Co.
# common packages
import numpy as np
import pickle as pkl
import argparse 
import os
import sys
sys.path.insert(0, '../')
from helper import *
from evaluation import *

from time import localtime, strftime
import time
import copy

# configuration parser
parser = argparse.ArgumentParser(description = "Configuration.")
parser.add_argument('--resultroot', type=str, default='../results/')
## datasets
parser.add_argument('--dataset', type=str, default='adult')
parser.add_argument('--dropoutmethod', type=str, default='bernoulli', choices=['bernoulli', 'gaussian'])
parser.add_argument('--drp_nmodel', type=int, default=10000)
parser.add_argument('--drp_max_ratio', type=float, default=0.2)
## ensemble
parser.add_argument('--ensemble_size', type=str, default='') 
parser.add_argument('--nensemble', type=int, default=100)
args = parser.parse_args()
configuration_dict = vars(args)

# fetch cuda device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
filename = args.dataset + '-' + args.dropoutmethod + '-dropout-' + str(args.drp_nmodel) + '-' + str(args.drp_max_ratio) + '-' + args.ensemble_size + '-' + str(args.nensemble) + '-ensemble'

log = open(args.resultroot + filename+'-log.txt', 'w+')
log.write('=== {} ===\n'.format(strftime("%Y-%m-%d-%H.%M.%S", time.localtime())))
log.write('Argument Summary\n')
for key in configuration_dict.keys():
    log.write(' {}: {}\n'.format(key, configuration_dict[key]))

log.write('Device: {}\n'.format(device))
log.flush()

## load base model for comparison
base_data_filename = [i for i in os.listdir(args.resultroot) if args.dataset+'-base' in i and '.npz' in i]
base_data = np.load(args.resultroot+base_data_filename[0])
base_y_test=base_data['y_test']
base_train_acc=base_data['train_acc']
base_test_acc=base_data['test_acc']
base_train_loss=base_data['train_loss']
base_test_loss=base_data['test_loss']
base_test_scores=base_data['test_scores']
base_train_time=base_data['train_time']

base_pred_y = base_test_scores.argmax(axis=1)

drp_data_filename = [i for i in os.listdir(args.resultroot) if args.dataset + '-' + args.dropoutmethod + '-' + args.method in i and str(args.drp_nmodel) + '-' + str(args.drp_max_ratio)+'.npz' in i]
drp_data = np.load(args.resultroot+drp_data_filename[0])
drp_all_test_scores=drp_data['all_test_scores'][0, :, :, :]
nsample = drp_all_test_scores.shape[1]
nclass = drp_all_test_scores.shape[2]


ensemble_size = np.array([int(i) for i in args.ensemble_size.split(',')])
nsize = len(ensemble_size)
ensemble_acc = np.zeros((nsize, args.nensemble))

vpr = np.zeros((nsize, nsample))
score_var = np.zeros((nsize, nsample))
rc = np.zeros((nsize, nsample))

amb = np.zeros((nsize, ))
disc = np.zeros((nsize, ))
disa = np.zeros((nsize, nsample))
disa_hat = np.zeros((nsize, nsample))

for i, size in enumerate(ensemble_size):
    log.write('{:3d}/{:3d} size = {:4d}\n'.format(i+1, nsize, size))
    log.flush()

    idx = np.random.choice(args.drp_nmodel, size=(args.nensemble, size), replace=False)

    ensemble_scores = np.zeros((args.nensemble, nsample, nclass))
    for j in range(args.nensemble):
        temp_scores = drp_all_test_scores[idx[j, :], :, :]
        ensemble_scores[j, :, :] = temp_scores.mean(axis=0) ## average of scores
        ensemble_decisions = ensemble_scores[j, :, :].argmax(axis=1)
        ensemble_acc[i, j] = (ensemble_decisions==base_y_test).mean()


    ensemble_score_y = score_of_y_multi_model(ensemble_scores, base_y_test)
    ## score-based
    vpr[i, :] = viable_prediction_range(ensemble_score_y)
    score_var[i, :] = score_variance(ensemble_score_y)
    rc[i, :] = rashomon_capacity(ensemble_scores)
    
    ## decision-based
    ensemble_decisions = ensemble_scores.argmax(axis=2)
    amb[i] = ambiguity(ensemble_decisions, base_pred_y)
    disc[i] = discrepancy(ensemble_decisions, base_pred_y)
    disa_hat[i, :] = disagreement_hat(ensemble_decisions)

savename = filename + '.npz'

np.savez_compressed(args.resultroot+savename,
                    ensemble_size=ensemble_size,
                    ensemble_acc=ensemble_acc,
                    vpr=vpr,
                    score_var=score_var,
                    rc=rc,
                    amb=amb,
                    disc=disc,
                    disa_hat=disa_hat
                    )

print('Finished!!!')
log.write('Finished!!!\n')
log.close()

