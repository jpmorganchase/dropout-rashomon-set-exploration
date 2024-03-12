
## Setup Conda Env
Note that, for mmcv, please install the version compatible to your cuda version. 
You can find the install guide [here](https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-with-pip)
```
conda create -n dropout_rashomon_det python==3.8.11
conda activate dropout_rashomon_det
pip install -r requirements.txt
pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout f78af7785ada87f1ced75a2313746e4ba3149760
pip install -e .
cp -f ../test.py tools/test.py
```

## Prepare MS COCO data and pretrained models
Please follow the [instructions](https://mmdetection.readthedocs.io/en/latest/user_guides/dataset_prepare.html) from mmdetection to setup MS COCO dataset and get pretrained models (YoloV3 and MaskRCNN) from their GitHub and put in `checkpoints` folder.
From `mmdetection` directory, download checkpoints.
```
mkdir -p checkpoints
cd checkpoints
wget https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_2x_coco/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth
wget https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_320_273e_coco/yolov3_d53_320_273e_coco-421362b6.pth
```

## Usage
Go back to `detecton` folder, use the `generate_run_all.py` to generate bash script to control type of dropout and dropout rate.
Then, copy the generated bash scripts and analysis python file (`predicitive_multiplicitiy_analysis.py`) into `mmdetection` folder to run detection results with different settings.
YoloV3 results are stored in `rashomon` folder while MaskRCNN results are located in `rashomon_maskrcnn` folder.
```
python3 generate_run_all.py
cp run_all_yolov3.sh mmdetection/
cp run_all_maskrcnn.sh mmdetection/
cp predicitive_multiplicitiy_analysis.py mmdetection/

cd mmdetection/
bash run_all_yolov3.sh
bash run_all_maskrcnn.sh
python3 predicitive_multiplicitiy_analysis.py yolov3
python3 predicitive_multiplicitiy_analysis.py maskrcnn
```