# SPDX-License-Identifier: Apache-2.0
# Copyright : J.P. Morgan Chase & Co.
import numpy as np

REPEAT=30

def main():
    
    with open('run_all.sh', 'w') as f:
        
        for dropout_type in ['no_drop', 'gaussian', 'bernoulli']:
            if dropout_type == 'gaussian':
                for dropout_rate in [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.045, 0.05]:
                    for _ in range(REPEAT):
                        # print(f"python tools/test.py configs/yolo/yolov3_d53_8xb8-320-273e_coco.py checkpoints/yolov3_d53_320_273e_coco-421362b6.pth --cfg-options test_evaluator.classwise=True --dropout_type {dropout_type} -p {dropout_rate} --work-dir ./rashomon/", file=f)
                        print(f"python tools/test.py configs/mask_rcnn/mask-rcnn_r50_fpn_2x_coco.py checkpoints/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth --cfg-options test_evaluator.classwise=True --dropout_type {dropout_type} -p {dropout_rate} --work-dir ./rashomon_maskrcnn/", file=f)
            if dropout_type == 'bernoulli':
                for dropout_rate in [5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 1e-4, 15e-4, 2e-4, 25e-4, 3e-4]:
                    for _ in range(REPEAT):
                        print(f"python tools/test.py configs/mask_rcnn/mask-rcnn_r50_fpn_2x_coco.py checkpoints/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth --cfg-options test_evaluator.classwise=True --dropout_type {dropout_type} -p {dropout_rate} --work-dir ./rashomon_maskrcnn/", file=f)
        
if __name__ == "__main__":
    main()