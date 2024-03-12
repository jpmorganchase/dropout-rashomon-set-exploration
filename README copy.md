This is the official Python implementation of the paper 
<a href="https://openreview.net/forum?id=Sf2A2PUXO3">Dropout-Based Rashomon Set Exploration for Efficient Predictive Multiplicity Estimation</a> (ICLR 2024):

This repository contains 3 different algorithms to explore the Rashomon set (re-training, dropout and adversarial weight perturbation (AWP)) and the estimation of 6 predictive multplicity metrics. 

## Installation
Download the code
```
git clone https://github.com/jpmorganchase/dropout-rashomon-set-exploration.git
cd dropout-rashomon-set-exploration
```

A suitable [conda](https://conda.io/) environment named `dropout-rashomon` can be created and activated with:
```
conda env create -f conda_env.yml
conda activate dropout-rashomon
pip install -r requirements.txt
```

* If you encounter the Runtime error: GET was unable to find an engine to execute, see [here](https://github.com/pytorch/pytorch/issues/102535) for a quick fix.

## Usage 
#### `synthetic/` contains the jupyter notebook to reproduce the results in Figure 1.
#### `uci-tabular/` contains the experiments for the UCI tabular datasets.
First, run `train-tabular.py` for different strategies to explore the Rashomon set; for example:
```
## Base Model
python3 train-tabular.py --dataset 'adult' --method 'base'

## Re-training Strategy (sampling)
python3 train-tabular.py --dataset 'adult' --method 'sampling' --sampling_nmodel 100 --nepoch 20
python3 train-tabular.py --dataset 'adult' --method 'sampling' --sampling_nmodel 100 --nepoch 25
python3 train-tabular.py --dataset 'adult' --method 'sampling' --sampling_nmodel 100 --nepoch 30
python3 train-tabular.py --dataset 'adult' --method 'sampling' --sampling_nmodel 100 --nepoch 35
python3 train-tabular.py --dataset 'adult' --method 'sampling' --sampling_nmodel 100 --nepoch 40
python3 train-tabular.py --dataset 'adult' --method 'sampling' --sampling_nmodel 100 --nepoch 45
python3 train-tabular.py --dataset 'adult' --method 'sampling' --sampling_nmodel 100 --nepoch 50
python3 train-tabular.py --dataset 'adult' --method 'sampling' --sampling_nmodel 100 --nepoch 55
python3 train-tabular.py --dataset 'adult' --method 'sampling' --sampling_nmodel 100 --nepoch 60

## Dropout Strategy
python3 train-tabular.py --dataset 'adult' --method 'dropout' --dropoutmethod 'bernoulli' --drp_nmodel 100 --drp_max_ratio 0.2
python3 train-tabular.py --dataset 'adult' --method 'dropout' --dropoutmethod 'gaussian' --drp_nmodel 100 --drp_max_ratio 0.6

## AWP Strategy (very time-consuming)
python3 train-tabular.py --dataset 'adult' --method 'awp'
```

Then run `../utils/compute_metrics.py` to evaluate the predictive multplicity metrics and the results will be saved in `results/`; for example:
```
python3 ../utils/compute_metrics.py --dataset 'adult' --method 'sampling' --sampling_nmodel 100 --epoch 20,25,30,35,40,45,50,55
python3 ../utils/compute_metrics.py --dataset 'adult' --method 'dropout' --dropoutmethod 'bernoulli' --drp_nmodel 100 --drp_max_ratio 0.2
python3 ../utils/compute_metrics.py --dataset 'adult' --method 'dropout' --dropoutmethod 'gaussian' --drp_nmodel 100 --drp_max_ratio 0.6
```

Similarly for dropout ensembles, run the following scripts:
```
## Ensemble Models
python3 train-tabular.py --dataset 'adult' --method 'dropout' --dropoutmethod 'bernoulli' --drp_nmodel 10000 --drp_max_ratio 0.2
python3 train-tabular.py --dataset 'adult' --method 'dropout' --dropoutmethod 'gaussian' --drp_nmodel 10000 --drp_max_ratio 0.6
python3 ../utils/ensemble.py --dataset 'adult' --dropoutmethod 'bernoulli' --drp_nmodel 10000 --drp_max_ratio 0.2 --ensemble_size 1,2,5,10,20,50,100 --nensemble 100
python3 ../utils/ensemble.py --dataset 'adult' --dropoutmethod 'gaussian' --drp_nmodel 10000 --drp_max_ratio 0.6  --ensemble_size 1,2,5,10,20,50,100 --nensemble 100
```

For model selection, run the following scripts:
```
## Model Selection
python3 train-model-selection.py --dataset 'adult' --nretraining 10  --dropoutmethod 'bernoulli' --drp_nmodel 100 --drp_max_ratio 0.2
python3 train-model-selection.py --dataset 'adult' --nretraining 10  --dropoutmethod 'gaussian' --drp_nmodel 100 --drp_max_ratio 0.6
python3 ../utils/compute_model_selection.py --dataset 'adult' --dropoutmethod 'bernoulli' --nretraining 10 --nepoch 100
python3 ../utils/compute_model_selection.py --dataset 'adult' --dropoutmethod 'gaussian' --nretraining 10 --nepoch 100
```

#### `vision/` contains the experiments for the vision datasets (CIFAR-10 and CIFAR-100).
Run `train-vision.py` for different strategies to explore the Rashomon set; for example:
```
## Base Model
python3 train-vision.py --dataset 'cifar10' --model 'vgg16' --method 'base' --nepoch 7

## Re-training Strategy (sampling)
python3 train-vision.py --dataset 'cifar10' --model 'vgg16' --method 'sampling' --sampling_nmodel 20 --nepoch 5
python3 train-vision.py --dataset 'cifar10' --model 'vgg16' --method 'sampling' --sampling_nmodel 20 --nepoch 6
python3 train-vision.py --dataset 'cifar10' --model 'vgg16' --method 'sampling' --sampling_nmodel 20 --nepoch 7
python3 train-vision.py --dataset 'cifar10' --model 'vgg16' --method 'sampling' --sampling_nmodel 20 --nepoch 8
python3 train-vision.py --dataset 'cifar10' --model 'vgg16' --method 'sampling' --sampling_nmodel 20 --nepoch 9

## Dropout Strategy
python3 train-vision.py --dataset 'cifar10' --model 'vgg16' --nepoch 7 --method 'dropout' --dropoutmethod 'bernoulli' --drp_nmodel 50 --ndrp 5 --drp_max_ratio 0.008
python3 train-vision.py --dataset 'cifar10' --model 'vgg16' --nepoch 7 --method 'dropout' --dropoutmethod 'gaussian' --drp_nmodel 50  --ndrp 5 --drp_max_ratio 0.1
```

Then run `../utils/compute_metrics.py` to evaluate the predictive multplicity metrics; for example:
```
python3 ../utils/compute_metrics.py --dataset 'cifar10' --model 'vgg16' --base_epoch 7 --method 'sampling' --sampling_nmodel 20 --epoch 5,6,7,8,9 --neps 6 --eps_max 0.05
python3 ../utils/compute_metrics.py --dataset 'cifar10' --model 'vgg16' --base_epoch 7 --method 'dropout' --dropoutmethod 'bernoulli' --drp_nmodel 50 --neps 6 --eps_max 0.05 --drp_max_ratio 0.008
python3 ../utils/compute_metrics.py --dataset 'cifar10' --model 'vgg16' --base_epoch 7 --method 'dropout' --dropoutmethod 'gaussian'  --drp_nmodel 50 --neps 6 --eps_max 0.05 --drp_max_ratio 0.1
```

#### `detection/` contains the codes and figure generation for the experiments of human detection. See [detection_readme](detection/README.md) for more information.
#### `utils/` is the main codebase, containing data loader, dropout methods, the AWP algorithm, and the computation of predictive multplicity metrics. 
#### `notebooks/` contains the jupyter notebooks to generate figures for the UCI tabular and CIFAR-10/-100 datasets by reading the evaluation results from `results/`.

## Citation
```
@article{hsu2024dropout,
  title={Dropout-Based Rashomon Set Exploration for Efficient Predictive Multiplicity Estimation},
  author={Hsu, Hsiang and Li, Guihong and Hu, Shaohan and Chen, Chun-Fu (Richard)},
  journal={International Conference on Learning Representations},
  year={2024}
}
```

## Contact
If you have any questions, please feel free to contact us through email (hsiang.hsu@jpmchase.com). Enjoy!