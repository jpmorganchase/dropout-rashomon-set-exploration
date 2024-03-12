
## Setup MMDetection
```
git clone https://github.com/open-mmlab/mmdetection.git
git checkout f78af7785ada87f1ced75a2313746e4ba3149760
cp test.py to mmdetection/tools/test.py
```

## Conda env
```
conda create -f environment.yml
pip install -r requirements.txt
```

## Usage
Use the `generate_run_all.py` to generate bash script to control type of dropout and dropout rate.
Then, use the generated bash script to run detection results with different settings.

Run `predicitive_multiplicitiy_analysis.py` to draw the figures and get numbers. (only consider the person class)