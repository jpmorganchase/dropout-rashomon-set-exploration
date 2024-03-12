# SPDX-License-Identifier: Apache-2.0
# Copyright : J.P. Morgan Chase & Co.
import pandas as pd
import numpy as np
import scipy as sp

##
from time import localtime, strftime
import time

## multicore accelerating
from itertools import islice
import multiprocessing
from multiprocessing import Pool

# helper functions
def score_of_y(scores, y):
    n = len(y)
    score_y = np.zeros((n,))
    for i in range(n):
        score_y[i] = scores[i, y[i]]
    return score_y

def score_of_y_multi_model(scores, y):
    nmodel, nsample, nclass = scores.shape[0], scores.shape[1], scores.shape[2]
    score_y = np.zeros((nmodel, nsample,))
    for i in range(nmodel): 
        for j in range(nsample):
            score_y[i, j] = scores[i, j, y[j]]
    return score_y

def get_Rashomon_models_dropout(loss, loss_tol, scores):
    # loss:     ndrp x nmodel
    # loss_tol: scalar
    # scores:   ndrp x nmodel x ntest x c
    ndrp = scores.shape[0]
    msk = loss <= loss_tol 

    msk_scores = []
    for i in range(ndrp):
        msk_scores.append(scores[i, msk[i, :], :, :])

    # print(len(msk_scores))
    all_msk_scores = msk_scores[0]
    for i in range(1, len(msk_scores)):
        # print(i)
        all_msk_scores = np.concatenate((all_msk_scores, msk_scores[i]), axis=0)

    return all_msk_scores

def get_Rashomon_models_sampling(loss, loss_tol, scores):
    # loss:     nmodel
    # loss_tol: scalar
    # scores:   nmodel x ntest x c
    msk = loss < loss_tol 

    return scores[msk, :, :]

# Multiplicity Metrics
## Decision Based
def ambiguity(decisions, y):
    nsample = decisions.shape[1]
    y = y.reshape((1, nsample))
    return np.any(decisions!=y, axis=0).mean()

def discrepancy(decisions, y):
    ## y could be the true label or the output of the base model
    nsample = decisions.shape[1]
    y = y.reshape((1, nsample))
    return ((decisions!=y).mean(axis=1)).max()

def disagreement_hat(decisions):
    # nmodel = decisions.shape[0]
    # nsample = decisions.shape[1]
    # nclass = scores.shape[2]
    mu = decisions.mean(axis=0)
    return 4*np.multiply(mu, 1-mu)

## Score Based
def rashomon_capacity(scores):
    nmodel = scores.shape[0]
    nsample = scores.shape[1]
    nclass = scores.shape[2]

    cores = multiprocessing.cpu_count() - 1
    it = iter(range(nsample))
    ln = list(iter(lambda: tuple(islice(it, 1)), ()))  # list of indices
    # compute in parallel
    with Pool(cores) as p:
        cvals = (p.map(blahut_arimoto, [scores[:, ix, :].reshape((nmodel, nclass)) for ix in ln]))
    capacity = np.array([v[0] for v in cvals])
    return capacity

def viable_prediction_range(scores):
    vpr = scores.max(axis=0)-scores.min(axis=0)
    return vpr

def score_variance(scores):
    # nmodel, nsample, nclass = scores.shape[0], scores.shape[1], scores.shape[2]

    return scores.var(axis=0)

def quantile_mean(v, q):
    assert q >= 0 and q <= 1
    q_value = np.quantile(v, q)
    return np.mean(v[v>=q_value])

def quantile_value(v, q):
    assert q >= 0 and q <= 1
    q_value = np.quantile(v, q)
    return q_value ## q = 0.5 => q_value = median

def blahut_arimoto(Pygw, log_base=2, epsilon=1e-12, max_iter=1e3):
    """
    Performs the Blahut-Arimoto algorithm to compute the channel capacity
    given a channel P_ygx.

    Parameters
    ----------
    Pygw: shape (m, c).
        transition matrix of the channel with m inputs and c outputs.
    log_base: int.
        base to compute the mutual information.
        log_base = 2: bits, log_base = e: nats, log_base = 10: dits.
    epsilon: float.
        error tolerance for the algorithm to stop the iterations.
    max_iter: int.
        number of maximal iteration.
    Returns
    -------
    Capacity: float.
        channel capacity, or the maximum information it can be transmitted
        given the input-output function.
    pw: array-like.
        array containing the discrete probability distribution for the input
        that maximizes the channel capacity.
    loop: int
        the number of iteration.
    resource: https://sites.ecse.rpi.edu/~pearlman/lec_notes/arimoto_2.pdf
    """
    ## check inputs
    # assert np.abs(Pygw.sum(axis=1).mean() - 1) < 1e-6
    # assert Pygw.shape[0] > 1

    m = Pygw.shape[0]
    c = Pygw.shape[1]
    Pw = np.ones((m)) / m
    for cnt in range(int(max_iter)):
        ## q = P_wgy
        q = (Pw * Pygw.T).T
        q = q / q.sum(axis=0)

        ## r = Pw
        r = np.prod(np.power(q, Pygw), axis=1)
        r = r / r.sum()

        ## stoppung criteria
        if np.sum((r - Pw) ** 2) / m < epsilon:
            break
        else:
            Pw = r

    ## compute capacity
    capacity = 0
    for i in range(m):
        for j in range(c):
            ## remove negative entries
            if r[i] > 0 and q[i, j] > 0:
                capacity += r[i] * Pygw[i, j] * np.log(q[i, j] / r[i])

    capacity = capacity / np.log(log_base)
    return capacity, r, cnt+1

## the data structure of awp is different from the rest
def score_of_y_multi_model_awp(scores, y):
    neps, nsample, nmodel, nclass = scores.shape[0], scores.shape[1], scores.shape[2], scores.shape[3]
    score_y = np.zeros((neps, nsample, nmodel, ))
    for i in range(neps): 
        for j in range(nsample):
            for k in range(nmodel):
                score_y[i, j, k] = scores[i, j, k, y[j]]
    return score_y

def rashomon_capacity_awp(scores):
    nsample = scores.shape[0]
    nmodel = scores.shape[1] ## equals to nclass
    nclass = scores.shape[2]

    cores = multiprocessing.cpu_count() - 1
    it = iter(range(nsample))
    ln = list(iter(lambda: tuple(islice(it, 1)), ()))  # list of indices
    # compute in parallel
    with Pool(cores) as p:
        cvals = (p.map(blahut_arimoto, [scores[ix, :, :].reshape((nmodel, nclass)) for ix in ln]))
    capacity = np.array([v[0] for v in cvals])
    return capacity

def viable_prediction_range_awp(scores):
    nsample = scores.shape[0]
    nmodel = scores.shape[1] 
    vpr = scores.max(axis=1)-scores.min(axis=1)
    return vpr

def score_variance_awp(scores):
    # nsample = scores.shape[0]
    # nmodel = scores.shape[1] 
    return scores.var(axis=1)

def ambiguity_awp(decisions, y):
    # decisions: nsample * nmodels
    nsample = decisions.shape[0]
    nmodel = decisions.shape[1]
    y = y.reshape((nsample,1))
    return np.any(decisions!=y, axis=1).mean()

def discrepancy_awp(decisions, y):
    ## y could be the true label or the output of the base model
    # nmodel = scores.shape[0]
    nsample = decisions.shape[0]
    nmodel = decisions.shape[1]
    y = y.reshape((nsample,1))
    return ((decisions!=y).mean(axis=0)).max()

def disagreement_hat_awp(decisions):
    mu = decisions.mean(axis=1)
    return 4*np.multiply(mu, 1-mu)

## for visualization
def read_eval_results(file):
    # results = np.load('../../../results/dropout/'+filename+'.npz')
    results = np.load(file+'.npz')

    eps_list = results['eps_list']
    vpr = results['vpr']
    score_var = results['score_var']
    rc = results['rc']
    amb = results['amb']
    disc = results['disc']
    disa_hat = results['disa_hat']

    vpr=np.sort(vpr, axis=1)
    score_var=np.sort(score_var, axis=1)
    rc=np.sort(2**rc, axis=1)
    disa_hat=np.sort(disa_hat, axis=1)

    output = {
        'eps_list': eps_list,
        'vpr': vpr,
        'score_var': score_var,
        'rc': rc,
        'amb': amb,
        'disc': disc,
        'disa_hat': disa_hat
    }

    return output

def read_base_results(file):
    # base_data = np.load('../../../results/dropout/'+datasetname+'-base.npz')
    results = np.load(file+'.npz')

    output = {
        'y_test': results['y_test'],
        'train_acc': results['train_acc'],
        'test_acc': results['test_acc'],
        'train_loss': results['train_loss'],
        'test_loss': results['test_loss'],
        'test_scores': results['test_scores'],
        'train_time': results['train_time']
    }

    return output

def read_drp_results(file):
    results = np.load(file+'.npz')

    output = {
        'y_test': results['y_test'],
        'train_acc': results['train_acc'],
        'train_loss': results['train_loss'],
        'test_acc': results['test_acc'],
        'test_loss': results['test_loss'],
        'drp_test_acc': results['drp_test_acc'],
        'drp_test_loss': results['drp_test_loss'],
        'all_test_scores': results['all_test_scores'],
        'inference_time': results['inference_time'],
        'drp_list': results['drp_list']
    }

    return output 