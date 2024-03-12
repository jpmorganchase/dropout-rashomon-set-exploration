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
parser.add_argument('--nretraining', type=int, default=10)
parser.add_argument('--nepoch', type=int, default=100) 

args = parser.parse_args()
configuration_dict = vars(args)

# fetch cuda device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

filename = args.dataset + '-' + str(args.nretraining) + '-' + str(args.nepoch) + '-' + args.dropoutmethod + '-model-selection-eval'

log = open(args.resultroot + filename+'-log.txt', 'w+')
log.write('=== {} ===\n'.format(strftime("%Y-%m-%d-%H.%M.%S", time.localtime())))
log.write('Argument Summary\n')
for key in configuration_dict.keys():
    log.write(' {}: {}\n'.format(key, configuration_dict[key]))

log.write('Device: {}\n'.format(device))
log.flush()


drp_data_filename = [i for i in os.listdir(args.resultroot) if args.dataset + '-' + str(args.nretraining) + '-' + str(args.nepoch) in i and args.dropoutmethod +  '-model-selection.npz' in i]
drp_data = np.load(args.resultroot+drp_data_filename[0])
drp_all_scores = drp_data['all_test_scores']
drp_list = drp_data['drp_list']
y_test = drp_data['y_test']
test_scores = drp_data['test_scores']
drp_test_acc = drp_data['drp_test_acc']
drp_test_loss = drp_data['drp_test_loss']


nmodel = drp_all_scores.shape[0]
ndrp =  drp_all_scores.shape[1]
ntest = drp_all_scores.shape[3]

vpr = np.zeros((nmodel, ndrp, ntest))
score_var = np.zeros((nmodel, ndrp, ntest))
rc = np.zeros((nmodel, ndrp, ntest))

amb = np.zeros((nmodel, ndrp,))
disc = np.zeros((nmodel, ndrp,))
disa = np.zeros((nmodel, ndrp, ntest))
disa_hat = np.zeros((nmodel, ndrp, ntest))

for i in range(nmodel):
    for j in range(ndrp):
        scores = drp_all_scores[i, j, :, :, :]
        score_y = score_of_y_multi_model(scores, y_test)

        ## score-based
        vpr[i, j, :] = viable_prediction_range(score_y)
        score_var[i, j, :] = score_variance(score_y)
        rc[i, j, :] = rashomon_capacity(scores)
        
        ## decision-based
        decisions = scores.argmax(axis=2)
        pred_y = test_scores[i, :, :].argmax(axis=1)
        amb[i, j] = ambiguity(decisions, pred_y)
        disc[i, j] = discrepancy(decisions, pred_y)
        disa[i, j, :] = disagreement(decisions, pred_y)
        disa_hat[i, j, :] = disagreement_hat(decisions)

        log.write('Model {:2d}, Drp: {:3d}/{:3d}\n'.format(i+1, j+1, ndrp))
        log.flush()

savename = filename + '.npz'

np.savez_compressed(args.resultroot+savename,
                    drp_list=drp_list,
                    drp_test_acc=drp_test_acc,
                    drp_test_loss=drp_test_loss,
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

