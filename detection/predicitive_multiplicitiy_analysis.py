# SPDX-License-Identifier: Apache-2.0
# Copyright : J.P. Morgan Chase & Co.
import pickle
import json
import os
import torch
import numpy as np
from mmdet.evaluation.functional.bbox_overlaps import bbox_overlaps
import matplotlib.pyplot as plt
from tqdm import tqdm
import concurrent.futures
import matplotlib.cm as cm

import sys
sys.path.insert(0, '../../')
from utils.evaluation import ambiguity, disagreement_hat, discrepancy, rashomon_capacity, viable_prediction_range, score_variance

ARCH=sys.argv[1]
IoU = 0.5
REF = {
    'yolov3': {
        'ap50': 0.696,
        'config': 'yolov3_d53_8xb8-320-273e_coco.py',
    },
    'maskrcnn': {
        'ap50': 0.823,
        'config': 'mask-rcnn_r50_fpn_2x_coco.py',
    },

}


def load_gt(annotation):
    with open(annotation) as f:
        gt = json.load(f)
        
    new_ann = []
    for tmp in gt['annotations']:
        if tmp['category_id'] == 1:
            tmp.pop('segmentation')
            new_ann.append(tmp)
            
    name_2_ann = {}
    for v in new_ann:
        name = f'{v["image_id"]:012d}.jpg'
        if name not in name_2_ann:
            name_2_ann[name] = []
            
        # update bbox
        v['bbox'] = [v['bbox'][0], v['bbox'][1], v['bbox'][0] + v['bbox'][2], v['bbox'][1] + v['bbox'][3]]
    
        name_2_ann[name].append(v)
                
    return name_2_ann

def load_results(pkl_file, log_file, name_2_ann):

    cache_file = os.path.join(os.path.dirname(pkl_file), 'cache.pkl')

    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                o = torch.load(cache_file)
                return o['name_2_results'], o['person_ap_50']
    except:
        pass

    person_ap_50 = 0.0
    with open(log_file, 'r') as f:
        for line in f.readlines():
            if '| person         |' in line:
                person_ap_50 = float(line.split('|')[3].strip())
                break
        
    name_2_results = {}
    with open (pkl_file, 'rb') as f:
        # try:
        results = pickle.load(f)
        filtered_results = []
        for r in results:
            if os.path.basename(r['img_path']) not in name_2_ann.keys():
                continue
            name_2_results[os.path.basename(r['img_path'])] = r.copy()
            new_pred_instances = {'bboxes': [], 'labels': [], 'scores': []}
            for bbox, label, score in zip(r['pred_instances']['bboxes'], r['pred_instances']['labels'], r['pred_instances']['scores']):
                if label == 0:
                    new_pred_instances['bboxes'].append(bbox.numpy())
                    new_pred_instances['labels'].append(label.numpy())
                    new_pred_instances['scores'].append(score.numpy())
            name_2_results[os.path.basename(r['img_path'])]['pred_instances'] = new_pred_instances
            filtered_results.append(r)
        # except:
        #     pass

    torch.save({'name_2_results': name_2_results, 'person_ap_50': person_ap_50}, cache_file)

    return name_2_results, person_ap_50

def load_all_results(src_dirs, name_2_ann):
    dmodels = {'ap50': [], 'res': []}

    if not isinstance(src_dirs, list):
        src_dirs = [src_dirs]

    jobs = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        for src_dir in src_dirs:
            for d in os.listdir(src_dir):
                if d == REF[ARCH]['config']:
                    continue

                pkl_file = os.path.join(src_dir, d, 'results.pkl')
                log_file = os.path.join(src_dir, d, f'{d}.log')

                if os.path.exists(pkl_file) and os.path.exists(log_file):
                    jobs.append(executor.submit(load_results, pkl_file, log_file, name_2_ann))
                else:
                    print(f"{pkl_file} or {log_file} do not exist.")

    for job in jobs:
        result, ap_50 = job.result()
        dmodels['ap50'].append(ap_50)
        dmodels['res'].append(result)

    return dmodels


def _get_score_table(name_2_ann, model_res, iou):
    out = []
    for name in name_2_ann.keys():
        gt_bboxes = np.asarray([v['bbox'] for v in name_2_ann[name]])
        det_bboxes = np.asarray(model_res[name]['pred_instances']['bboxes'])
        det_scores = np.asarray(model_res[name]['pred_instances']['scores'])

        overlap = bbox_overlaps(gt_bboxes, det_bboxes)
        if overlap.shape[-1] < 1: # no detection.
            filtered_scores = np.zeros((gt_bboxes.shape[0]))
        else:
            max_idx = np.argmax(overlap, axis=1)
            filtered_scores = det_scores[max_idx]
            
            # update the scores to zeros for those iou below threshold
            maximum = np.max(overlap, axis=1)
            below_iou_idx = np.where(maximum < iou)[0]
            filtered_scores[below_iou_idx] = 0.0
        
            if gt_bboxes.shape[0] != len(filtered_scores):
                print(".....")
        out.extend(filtered_scores.tolist())
    return out

def get_score_table(name_2_ann, num_pred, dmodels, iou=0.5):
    num_models = len(dmodels['res'])
    score_table = np.zeros((num_models, num_pred, 2))


    jobs = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        for i in range(num_models):
            jobs.append(executor.submit(_get_score_table, name_2_ann, dmodels['res'][i], iou))

    for i, job in enumerate(jobs):
        out = np.asarray(job.result())
        score_table[i, ..., 1] = out
        score_table[i, ..., 0] = 1.0 - score_table[i, ..., 1]

    return score_table

def categorize_models_by_ap_loss(dmodels, score_table, loss_step=None):
    # categorize by ap50 difference
    dmodels['ap50'] = REF[ARCH]['ap50'] - np.asarray(dmodels['ap50'])
    # dmodels['res'] = np.asarray(dmodels['res'])
    _, hist_edges = np.histogram(np.asarray(dmodels['ap50']), bins=10)
    score_tables = {}
    if loss_step is not None:
        hist_edges = loss_step
    else:
        hist_edges = hist_edges[1:]

    for h in hist_edges:
        idx = np.where(dmodels['ap50'] <= h)[0]
        score_tables[h] = score_table[idx]
    return score_tables


def draw_all(gau_result, bern_result):
    fig, ax = plt.subplots(3, 4, figsize=(14, 5), gridspec_kw={'height_ratios': [1,2,2]})
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    for eps, res in gau_result.items():
        n = len(res['vpr'])
        break
    cum = np.arange(1, n+1)/n

    colors = cm.ocean(np.linspace(0, 1, len(gau_result.keys())+1))
    i = 0
    for eps, res in gau_result.items():

        ax[1, 0].plot(bern_result[eps]['vpr'], cum, color=colors[i])
        ax[1, 1].plot(bern_result[eps]['score_var'], cum, color=colors[i])
        ax[1, 2].plot(bern_result[eps]['rc'], cum, color=colors[i])
        ax[1, 3].plot(bern_result[eps]['disa_hat'], cum, color=colors[i], label=r'$\epsilon$ = {:.3f}'.format(eps))
        
        ax[2, 0].plot(res['vpr'], cum, color=colors[i])
        ax[2, 1].plot(res['score_var'], cum, color=colors[i])
        ax[2, 2].plot(res['rc'], cum, color=colors[i])
        ax[2, 3].plot(res['disa_hat'], cum, color=colors[i])

        i += 1        
    ax[2, 0].set_xlabel('Viable Prediction Range')
    ax[2, 1].set_xlabel('Score Variance')
    ax[2, 2].set_xlabel('Rashomon Capacity')
    ax[2, 3].set_xlabel('Disagreement')

    ax[1, 0].set_ylabel(r'${\bf Bernoulli}$' '\n' 'CDF of Samples')
    ax[2, 0].set_ylabel(r'${\bf Gaussian}$' '\n' 'CDF of Samples')
    ax[1, 2].legend(bbox_to_anchor=(0.5, 1.8), ncol=4, title='Rashomon Parameter')

    ax[0, 0].axis('off')
    ax[0, 1].axis('off')
    ax[0, 2].axis('off')
    ax[0, 3].axis('off')

    plt.tight_layout()
    plt.savefig(f'mscoco-{ARCH}-IoU{IoU:.1f}-scores.png', format='png', dpi=300, bbox_inches='tight')

def draw_decision_base(gau_res_, bern_res_):

    gau_res = {'eps_list': [],
          'amb': [],
          'disc': []}

    bern_res = {'eps_list': [],
            'amb': [],
            'disc': []}

    for eps, v in gau_res_.items():
        gau_res['eps_list'].append(eps)
        gau_res['amb'].append(v['ambiguity'])
        gau_res['disc'].append(v['discrepancy'])
        

    for eps, v in bern_res_.items():
        bern_res['eps_list'].append(eps)
        bern_res['amb'].append(v['ambiguity'])
        bern_res['disc'].append(v['discrepancy'])


    fig, ax = plt.subplots(1, 2, figsize=(7, 2.5))
    colors = ['maroon', 'olive', 'orange', 'darkgoldenrod']
    ms = 3

    # NOTE: this part might be need to adjust the figure locations
    ax[0].plot(bern_res['eps_list'], bern_res['amb'], markersize=ms, marker='s', color=colors[1])
    ax[0].plot(gau_res['eps_list'], gau_res['amb'], markersize=ms, marker='^', color=colors[2])
    ax[0].set_xlabel(r'Rashomon Parameter $\epsilon$')
    ax[0].set_ylabel('Ambiguity')

    ax[1].plot(bern_res['eps_list'], bern_res['disc'], label='Bernoulli', markersize=ms, marker='s', color=colors[1])
    ax[1].plot(gau_res['eps_list'], gau_res['disc'], label='Gaussian', markersize=ms, marker='^', color=colors[2])
    ax[1].set_xlabel(r'Rashomon Parameter $\epsilon$')
    ax[1].set_ylabel('Discrepancy');
    ax[1].legend(loc='best', fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(f'mscoco-{ARCH}-IoU{IoU:.1f}-decision-based.png', format='png', dpi=300)


def _main(src_dirs, name_2_ann, num_pred, loss_step=None):
    
    print(f"Loading Results from {src_dirs}")
    dmodels = load_all_results(src_dirs, name_2_ann)
    print("Getting Score Tables for all models...")
    score_table = get_score_table(name_2_ann, num_pred, dmodels, IoU)
    print("Categorizing by AP_50 loss...")
    score_tables = categorize_models_by_ap_loss(dmodels, score_table, loss_step)

    for ap_loss, res in score_tables.items():
        print(f"{ap_loss:.4f}, {res.shape[0]}")

    pm_res = {}
    pm_res_all = {}

    print("Starting to evaluate the PM...")
    for ap_loss, score_table in score_tables.items():
        print(f"AP loss is {ap_loss:.4f}, {score_table.shape[0]} models")

        p50 = (score_table.shape[1] + 1) // 2
        p80 = int(score_table.shape[1] * 0.8 + 0.5)

        vpr_raw_sorted = np.sort(viable_prediction_range(score_table[:,:, 1]))
        vpr = vpr_raw_sorted[p50]

        sv_raw_sorted = np.sort(score_variance(score_table[:,:, 1]))
        sv = sv_raw_sorted[p50]

        rc_raw_sorted = np.sort(rashomon_capacity(score_table))
        rc_raw_sorted_power = np.power(2, rc_raw_sorted)
        rc = rc_raw_sorted_power[p50]
        
        disa_raw_sorted = np.sort(disagreement_hat(np.argmax(score_table, axis=-1), np.ones(score_table.shape[1])))
        disa = disa_raw_sorted[p80]

        am = ambiguity(np.argmax(score_table, axis=-1), np.ones(score_table.shape[1]))
        disc = discrepancy(np.argmax(score_table, axis=-1), np.ones(score_table.shape[1]))

        pm_res_all[ap_loss] = {
            'vpr': vpr_raw_sorted,
            'score_var': sv_raw_sorted,
            'rc': rc_raw_sorted_power,
            'disa_hat': disa_raw_sorted,
            'ambiguity': am,
            'discrepancy': disc
        }

        pm_res[ap_loss] = {
            'vpr': vpr,
            'sv': sv,
            'rc': rc,
            'disagreements': disa,
            'ambiguity': am,
            'discrepancy': disc
        }

    return pm_res, pm_res_all

def main():
    print("Loading GT...")
    name_2_ann = load_gt('./data/coco/annotations/instances_val2017.json')
    num_pred = sum([len(v) for _, v in name_2_ann.items()])

    if ARCH == 'yolov3':
        loss_step = np.linspace(0.005, 0.012, num=8)
    else:
        loss_step = np.linspace(0.005, 0.012, num=8)

    if ARCH == 'yolov3':
        src_dirs = [
            'rashomon/gaussian_0.0300000/', 
            'rashomon/gaussian_0.0400000/', 
            'rashomon/gaussian_0.0500000/', 
            'rashomon/gaussian_0.0600000/',
            'rashomon/gaussian_0.0700000/', 
            'rashomon/gaussian_0.0800000/',
            'rashomon/gaussian_0.0900000/',
            'rashomon/gaussian_0.1000000/',
        ]
    else:
        src_dirs = [
            'rashomon_maskrcnn/gaussian_0.0100000',
            'rashomon_maskrcnn/gaussian_0.0150000',
            'rashomon_maskrcnn/gaussian_0.0200000',
            'rashomon_maskrcnn/gaussian_0.0250000',
            'rashomon_maskrcnn/gaussian_0.0300000',
            'rashomon_maskrcnn/gaussian_0.0350000',
            'rashomon_maskrcnn/gaussian_0.0400000',
            'rashomon_maskrcnn/gaussian_0.0450000',
            'rashomon_maskrcnn/gaussian_0.0500000',
            'rashomon_maskrcnn/gaussian_0.0550000',
            'rashomon_maskrcnn/gaussian_0.0600000',
            'rashomon_maskrcnn/gaussian_0.0650000',
            'rashomon_maskrcnn/gaussian_0.0700000',
        ]
    gau_res, gau_res_all = _main(src_dirs, name_2_ann, num_pred, loss_step)
    
    if ARCH == 'yolov3':
        src_dirs = [
            'rashomon/bernoulli_0.0000500/',
            'rashomon/bernoulli_0.0000600/',
            'rashomon/bernoulli_0.0000700/',
            'rashomon/bernoulli_0.0000800/', 
            'rashomon/bernoulli_0.0000900/', 
            'rashomon/bernoulli_0.0001000/',
        ]
    else:
        src_dirs = [
            'rashomon_maskrcnn/bernoulli_0.0000500',
            'rashomon_maskrcnn/bernoulli_0.0000600',
            'rashomon_maskrcnn/bernoulli_0.0000700',
            'rashomon_maskrcnn/bernoulli_0.0000800',
            'rashomon_maskrcnn/bernoulli_0.0000900',
            'rashomon_maskrcnn/bernoulli_0.0001000',
            'rashomon_maskrcnn/bernoulli_0.0002000',
            'rashomon_maskrcnn/bernoulli_0.0003000',
        ]

    bern_res, bern_res_all = _main(src_dirs, name_2_ann, num_pred, loss_step)
    
    draw_all(gau_res_all, bern_res_all)
    draw_decision_base(gau_res, bern_res)

    torch.save(
        {
            'gau_all': gau_res_all,
            'gau': gau_res,
            'bern_res_all': bern_res_all,
            'bern_res': bern_res,
        },
        f'mscoco-{ARCH}-IoU{IoU:.1f}-bern-gauss.pkl'
    )

if __name__ == "__main__":
    main()