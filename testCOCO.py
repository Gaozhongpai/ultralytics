import os, os.path as osp
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics.yolo.data.dataloaders.pigeneral import LOGGER, check_dataset_v2, colorstr
from ultralytics.yolo.data.dataloaders.piloader import create_dataloader
from pathlib import Path


data = "ultralytics/yolo/data/datasets/JointBP_BodyHands.yaml"
data = check_dataset_v2(data)  # check
annot_part = osp.join(data['path'], data['val_annotations_part'])
coco = COCO(annot_part)
json_path_part_coco = "runs/detect/BodyHand3/weights/val_best_c0.05_i0.6_bodypart_coco.json"
result = coco.loadRes(json_path_part_coco)
eval = COCOeval(coco, result, iouType='bbox')

imgsz = 640
batch_size = 8
gs = 32
single_cls = False
task = "val"
pad = 0
rect = True 

dataloader = create_dataloader(data["val"], imgsz, batch_size, gs, 
            pad=pad, rect=rect, quad=True, prefix=colorstr(f'{task}: '))[0]
# eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
eval.params.imgIds = [int(Path(x).stem.split("_")[-1]) for x in dataloader.dataset.img_files]
eval.evaluate()
eval.accumulate()
eval.summarize()