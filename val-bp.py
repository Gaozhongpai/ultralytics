import argparse
import json
import os, os.path as osp
import sys
from pathlib import Path

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add kapao/ to path

import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.data.dataloaders.piloader import create_dataloader, letterbox
from ultralytics.yolo.data.dataloaders.pigeneral import LOGGER, check_dataset_v2, check_file, check_img_size, \
    non_max_suppression, scale_boxes, set_logging, colorstr, xyxy2xywh
from ultralytics.yolo.utils.torch_utils import select_device, time_sync
import tempfile
import cv2
import pickle

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics.yolo.utils.metrics import bbox_ioa

from ultralytics.yolo.utils.bp_eval import body_part_association_evaluation

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

def cal_inside_iou(bigBox, smallBox):  # body_box, part_box
    # calculate small rectangle inside big box ratio, calSmallBoxInsideRatio
    [Ax0, Ay0, Ax1, Ay1] = bigBox[0:4]
    [Bx0, By0, Bx1, By1] = smallBox[0:4]
    W = min(Ax1, Bx1) - max(Ax0, Bx0)
    H = min(Ay1, By1) - max(Ay0, By0)
    if W <= 0 or H <= 0:
        return 0
    else:
        areaA = (Ax1 - Ax0)*(Ay1 - Ay0)
        areaB = (Bx1 - Bx0)*(By1 - By0)
        crossArea = W * H
        # return crossArea/(areaA + areaB - crossArea)
        return crossArea/areaB  # range [0, 1]
    

def post_process_batch(data, imgs, paths, shapes, body_dets, part_dets):

    batch_bboxes, batch_points, batch_scores, batch_imgids = [], [], [], []
    batch_parts_dict = {}
    img_indexs = []
    color = Colors()
    num = 1
    if data['dataset'] == "BodyHands":
        num = num + 1
    # process each image in batch
    for si, (bdet, pdet) in enumerate(zip(body_dets, part_dets)):
        nbody, npart = bdet.shape[0], pdet.shape[0]
        if npart and nbody:  # one batch
            path, shape = Path(paths[si]) if len(paths) else '', shapes[si][0]
            
            # img_id = int(osp.splitext(osp.split(path)[-1])[0]) if path else si
            if data['dataset'] == "CityPersons" or data['dataset'] == "CrowdHuman" or data['dataset'] == "BodyHands":
                img_id = int(osp.splitext(osp.split(path)[-1])[0].split("_")[-1]) if path else si
                
            ## part 
            bboxes = scale_boxes(imgs[si].shape[1:], pdet[:, :4], shape, shapes[si][1]).cpu().numpy()
            bp_bboxes = scale_boxes(imgs[si].shape[1:], pdet[:, -4:], shape, shapes[si][1]).cpu().numpy()
            px1, py1, px2, py2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
            conf, cls = pdet[:, 4].cpu().numpy(), pdet[:, 5].cpu().numpy()
            p_xc, p_yc = np.mean((px1, px2), axis=0), np.mean((py1, py2), axis=0)
            
            # img = cv2.imread(paths[si])
            # for k, b_bbox in enumerate(bboxes):
            #     c = color(int(k*color.n/len(bboxes)))
            #     bx0, by0, bx1, by1 = b_bbox
            #     cv2.rectangle(img, (int(bx0), int(by0)), (int(bx1), int(by1)), c, thickness=4)
            #     px, py = points[k, 0], points[k, 1]
            #     cv2.line(img, (int(bx0), int(by0)), (int(px), int(py)), c, thickness=4)
            # cv2.imwrite("test.jpg", img)
            
            ## body 
            scores = bdet[:, 4].cpu().numpy()  # body detection score
            b_bboxes = scale_boxes(imgs[si].shape[1:], bdet[:, :4].clone(), shape, shapes[si][1]).cpu().numpy()
            matched_part_ids = [-1 for i in range(bp_bboxes.shape[0])]  # points shape is n*c*7, add in 2022-12-09
            
            part_pts = np.zeros((nbody, num, 7))
            
            batch_parts_dict[str(img_id)] = []
            for id, bp_bbox in enumerate(bp_bboxes):
                iou = bbox_ioa(bp_bbox[None], b_bboxes)[0]
                sorted_indices = np.argsort(iou)[::-1]
                # pt_match = np.argmin(dist)
                i = 0
                pt_match = sorted_indices[i]
                count = np.count_nonzero(matched_part_ids == pt_match)
                batch_parts_dict[str(img_id)].append([px1[id], py1[id], px2[id], py2[id], conf[id], cls[id]])
                while (count > num-1) and (i < nbody-1): ## maximum two hands
                    i = i + 1
                    pt_match = sorted_indices[i]
                    count = np.count_nonzero(matched_part_ids == pt_match)
                if count < num and (p_xc[id]> b_bboxes[pt_match][0] and ## hand is within body 
                                  p_xc[id]< b_bboxes[pt_match][2] and 
                                  p_yc[id]> b_bboxes[pt_match][1] and 
                                  p_yc[id]< b_bboxes[pt_match][3]):
                    part_pts[pt_match, count] = [p_xc[id], p_yc[id], conf[id], px1[id], py1[id], px2[id], py2[id]]
                matched_part_ids[id] = pt_match            
            # print(matched_part_ids)
            
            img = cv2.imread(paths[si])
            for k, b_bbox in enumerate(b_bboxes):
                c = color(int(k*color.n/len(b_bboxes)))
                bx0, by0, bx1, by1 = b_bbox
                cv2.rectangle(img, (int(bx0), int(by0)), (int(bx1), int(by1)), c, thickness=4)
                for bbox in part_pts[k]:
                    x0, y0, x1, y1 = bbox[3:]
                    if x0+y0+x1+y1!=0:
                        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), c, thickness=4)
                        cv2.line(img, (int(x0), int(y0)), (int(bx0), int(by0)), c, thickness=4)
            
            # for k, b_bbox in enumerate(bboxes):
            #     c = color(int(k*color.n/len(b_bboxes)))
            #     bx0, by0, bx1, by1 = b_bbox
            #     px, py = points[k, 0], points[k, 1]
            #     cv2.line(img, (int(bx0), int(by0)), (int(px), int(py)), c, thickness=4)
            Path("./runs/debug/").mkdir(parents=True, exist_ok=True)
            cv2.imwrite("./runs/debug/"+Path(paths[si]).stem+".jpg", img)
                        
            batch_bboxes.extend(b_bboxes)
            batch_points.extend(part_pts)
            batch_scores.extend(scores)
            batch_imgids.extend([img_id] * len(scores))
            
            img_indexs.append(si)
        else:
            print("This image has no object detected!")
        
    return batch_bboxes, batch_points, batch_scores, batch_imgids, batch_parts_dict, img_indexs
    
            
@torch.no_grad()
def run(opt, data,
        weights=None,  # model.pt path(s)
        batch_size=16,  # batch size
        imgsz=1280,  # inference size (pixels)
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        single_cls=False,  # treat as single-class dataset
        conf_thres=0.01,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        scales=[1],
        flips=[None],
        rect=False,
        half=True,  # use FP16 half-precision inference
        model=None,
        dataloader=None,
        compute_loss=None,
        pad=0,
        json_name='',
        no_trace=False,
        augment=False,
        ):

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device
    else:  # called directly
        device = select_device(device, batch=batch_size)

        dnn=False
        half=True
        # Load model
        
        model = YOLO('yolov8m.yaml').load(weights)
        model = AutoBackend(model.model, device=device, dnn=False, data=None, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine            
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        # Data
        data = check_dataset_v2(data)  # check
        
    if data['dataset'] == "CityPersons":  
        # data['dist_thre'] = 200  # the largest dist threshold for matching, large than it will not be replaced
        data['conf_thres'] = 0.01  # the larger conf threshold for filtering body detection proposals
        data['iou_thres'] = 0.6  # the smaller iou threshold for filtering body detection proposals
        data['conf_thres_part'] = 0.02  # the larger conf threshold for filtering body-part detection proposals
        data['iou_thres_part'] = 0.3  # the smaller iou threshold for filtering body-part detection proposals
    if data['dataset'] == "CrowdHuman" or data['dataset'] == "BodyHands":
        # data['dist_thre'] = 100
        data['conf_thres'] = 0.15  # CrowdHuman and BodyHands have more dense instance labels
        data['iou_thres'] = 0.6
        data['conf_thres_part'] = 0.25  # CrowdHuman and BodyHands have more dense instance labels
        data['iou_thres_part'] = 0.6
        
    data['match_iou_thres'] = 0.6  # whether a body-part in matched with one body bbox
    
    # Half
    half &= device.type != 'cpu'  # half precision only supported on CUDA
        
    # Configure
    model.eval()
    nc = int(data['nc'])  # number of classes
    gs = max(int(model.stride), 32)  # grid size (max stride)
    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], data["labels"], imgsz, batch_size, gs, 
                                       pad=pad, rect=rect, quad=True, prefix=colorstr(f'{task}: '))[0]

    color = Colors()
    seen = 0
    mp, mr, map50, mAP, mAP_part, map50_part, t0, t1, t2 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(4, device=device)
    json_dump, json_dump_part_coco, json_dump_part_mr = [], [], []
    
    pbar = tqdm(dataloader, desc='Processing {} images'.format(task))
    for batch_i, (imgs, targets, paths, shapes) in enumerate(pbar):
        t_ = time_sync()
        imgs = imgs.to(device, non_blocking=True)
        # imgs_ori = imgs.clone()
        imgs = imgs.half() if half else imgs.float()  # uint8 to fp16/32
        imgs /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = imgs.shape  # batch size, channels, height, width
        t = time_sync()
        t0 += t - t_

        # Run model
        out = model(imgs, augment=False)
        t1 += time_sync() - t

        # Compute loss
        train_out = None
        if train_out:  # only computed if no scale / flipping
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls, bpl

        t = time_sync()
        
        # Run NMS
        # left_out = non_max_suppression(out, conf_thres, iou_thres, 
            # multi_label=False, agnostic=single_cls, num_offsets=data['num_offsets'])
        # body_dets = [d[d[:, 5] == 0] for d in left_out]  # [xyxy, conf, cls, part_points], cls = 0
        # part_dets = [d[d[:, 5] > 0] for d in left_out]  # [xyxy, conf, cls, part_points], cls = 1 or larger numbers
        
        body_dets = non_max_suppression(out, data['conf_thres'], data['iou_thres'], classes=[0],
            multi_label=False, agnostic=single_cls)
        part_dets = non_max_suppression(out, data['conf_thres_part'], data['iou_thres_part'], 
            # classes=list(range(1, 1 + data['num_offsets']//2)),
            classes=list(range(1, data['nc'])),
            multi_label=False, agnostic=single_cls)
        
        
        # Post-processing of body and part detections
        bboxes, points, scores, imgids, parts_dict, img_indexs = post_process_batch(
            data, imgs, paths, shapes, body_dets, part_dets)

        t2 += time_sync() - t
        seen += len(imgs)
        
        for i, (bbox, point, score, img_id) in enumerate(zip(bboxes, points, scores, imgids)):
            bbox_new = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]  # [x0, y0, x1, y1] --> [x0, y0, w, h]

            # https://github.com/AibeeDetect/BFJDet/tree/main/eval_cp
            if data['dataset'] == "CityPersons" or data['dataset'] == "CrowdHuman":  # data['num_offsets'] is 2
                f_score, f_bbox = point[0][2], point[0][3:]  # bbox format [x1, y1, x2, y2]
                f_bbox = [f_bbox[0], f_bbox[1], f_bbox[2]-f_bbox[0], f_bbox[3]-f_bbox[1]]
                f_bbox = f_bbox if f_score != 0 else [0, 0, 1, 1]  # this format is defined in BFJDet 
                
                json_dump.append({
                    'image_id': img_id,
                    'category_id': 1,  # only one class 'person'
                    'bbox': [round(float(t), 3) for t in bbox_new],
                    'score': round(float(score), 3),  # person body score
                    'f_bbox': [round(float(t), 3) for t in f_bbox],  # the single bbox of body part (face or head)
                    'f_score': round(float(f_score), 3),  # the score of body part (face or head)
                })
            
                # [x0, y0, x1, y1] = bbox
                # cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), thickness=2)
                # [px0, py0, px1, py1] = f_bbox
                # if px0 != 0 and py0 != 0:
                    # cv2.rectangle(img, (int(px0), int(py0)), (int(px1), int(py1)), (0, 255, 0), thickness=2)
                    # cv2.line(img, (int(x0), int(y0)), (int(px0), int(py0)), (255,255,0), thickness=2)
                # cv2.imwrite("./debug/"+Path(paths[img_indexs[i]]).stem+".jpg", img)
            
            if data['dataset'] == "BodyHands":  # data['num_offsets'] is 4, BodyHands does not label left-right
                lh_score, lh_bbox = point[0][2], point[0][3:]  # hand1 part, bbox format [x1, y1, x2, y2]
                lh_bbox = [lh_bbox[0], lh_bbox[1], lh_bbox[2]-lh_bbox[0], lh_bbox[3]-lh_bbox[1]]
                lh_bbox = lh_bbox if lh_score != 0 else [0, 0, 1, 1]  # this format is defined in BFJDet
                
                rh_score, rh_bbox = point[1][2], point[1][3:]  # hand2 part, bbox format [x1, y1, x2, y2]
                rh_bbox = [rh_bbox[0], rh_bbox[1], rh_bbox[2]-rh_bbox[0], rh_bbox[3]-rh_bbox[1]]
                rh_bbox = rh_bbox if rh_score != 0 else [0, 0, 1, 1]  # this format is defined in BFJDet 
            
                # print(img.shape)
                # c = color(i%20)
                # [x0, y0, x1, y1] = bbox
                # cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), c, thickness=4)
                # [px0, py0, px1, py1] = point[0][3:]
                # if px0 != 0 and py0 != 0:
                #     cv2.rectangle(img, (int(px0), int(py0)), (int(px1), int(py1)), c, thickness=4)
                #     cv2.line(img, (int(x0), int(y0)), (int(px0), int(py0)), c, thickness=4)
                # [px0, py0, px1, py1] = point[1][3:] 
                # if px0 != 0 and py0 != 0:
                #     cv2.rectangle(img, (int(px0), int(py0)), (int(px1), int(py1)), c, thickness=4)
                #     cv2.line(img, (int(x0), int(y0)), (int(px0), int(py0)), c, thickness=4)
                # cv2.imwrite("./debug/"+Path(paths[img_indexs[i]]).stem+".jpg", img)
                
                json_dump.append({
                    'image_id': img_id,
                    'category_id': 1,  # only one class 'person'
                    'bbox': [round(float(t), 3) for t in bbox_new],
                    'score': round(float(score), 3),  # person body score
                    'h1_bbox': [round(float(t), 3) for t in lh_bbox],  # the single bbox of body hand1 part
                    'h1_score': round(float(lh_score), 3),  # the score of body part (hand1)
                    'h2_bbox': [round(float(t), 3) for t in rh_bbox],  # the single bbox of body hand2 part
                    'h2_score': round(float(rh_score), 3),  # the score of body part (hand2)
                })
        
        imgids_rmdup = list(set(imgids))
        for img_id in imgids_rmdup:
            part_bbox_list = parts_dict[str(img_id)]
            for part_bbox in part_bbox_list:
                [x1, y1, x2, y2, conf, cls] = part_bbox
                json_dump_part_coco.append({
                    'image_id': img_id,
                    'category_id': int(cls),  # class of body part, e.g., [1,] for 'head' or 'face', [1,2] for 'hands'
                    'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],  # [x0, y0, w, h]
                    'score': float(conf),  # using person body score as body part score
                })
                json_dump_part_mr.append({
                    'image_id': img_id,
                    'category_id': int(cls+1),  # class of body part, e.g., [2,] for 'head' or 'face', [2,3] for 'hands'
                    'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],  # [x0, y0, w, h]
                    'score': float(conf),  # using person body score as body part score
                })
                    
        # if batch_i > 2: break  # for prediction results debugging
        
        
    if not training:  # save json
        save_dir, weights_name = osp.split(weights)
        if not json_name:
            json_name = '{}_{}_c{}_i{}.json'.format(
                task, osp.splitext(weights_name)[0], data['conf_thres'], data['iou_thres'])
        else:
            if not json_name.endswith('.json'):
                json_name += '.json'
        json_path = osp.join(save_dir, json_name)
    else:
        tmp = tempfile.NamedTemporaryFile(mode='w+b')
        json_path = tmp.name + '.json'
    json_path_part_coco = json_path[:-5]+"_bodypart_coco.json"
    json_path_part_mr = json_path[:-5]+"_bodypart_mr.json"
    
    with open(json_path, 'w') as f:
        json.dump(json_dump, f)
    with open(json_path_part_coco, 'w') as f:
        json.dump(json_dump_part_coco, f)
    with open(json_path_part_mr, 'w') as f:
        json.dump(json_dump_part_mr, f)

    if len(json_dump) == 0:
        error_list = [0, 0, 0]
        return (mp, mr, map50, mAP, map50_part, mAP_part, *(loss.cpu() / len(dataloader)).tolist()), np.zeros(nc), t, error_list

    if task in ('train', 'val'):
        print("###### person bbox mAP:", len(json_dump))
        if len(json_dump) != 0:
            annot = osp.join(data['path'], data['{}_annotations'.format(task)])
            coco = COCO(annot)
            result = coco.loadRes(json_path)
            eval = COCOeval(coco, result, iouType='bbox')
            # eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.params.imgIds = [int(Path(x).stem.split("_")[-1]) for x in dataloader.dataset.img_files]
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            mAP, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
            
        print("###### bodypart bbox mAP:", len(json_dump_part_coco))
        if len(json_dump_part_coco) != 0:
            annot_part = osp.join(data['path'], data['{}_annotations_part'.format(task)])
            coco = COCO(annot_part)
            result = coco.loadRes(json_path_part_coco)
            eval = COCOeval(coco, result, iouType='bbox')
            # eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.params.imgIds = [int(Path(x).stem.split("_")[-1]) for x in dataloader.dataset.img_files]
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            mAP_part, map50_part = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        
        if data['dataset'] == "CityPersons":
            if len(json_dump) != 0 and len(json_dump_part_mr) != 0:
                MR_body_list, MR_part_list, mMR_list, MR_body, MR_part, mMR = body_part_association_evaluation(
                    json_path, json_path_part_mr, data)
            else:
                MR_body_list, MR_part_list, mMR_list = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
                MR_body, MR_part, mMR = 0, 0, 0

            print("[MR_body_list]: Reasonable: %.3f, Bare: %.3f, Partial: %.3f, Heavy: %.3f"%(
                MR_body_list[0], MR_body_list[1], MR_body_list[2], MR_body_list[3] ))
            print("[MR_part_list]: Reasonable: %.3f, Bare: %.3f, Partial: %.3f, Heavy: %.3f"%(
                MR_part_list[0], MR_part_list[1], MR_part_list[2], MR_part_list[3] ))
            print("[mMR_all_list]: Reasonable: %.3f, Bare: %.3f, Partial: %.3f, Heavy: %.3f"%(
                mMR_list[0], mMR_list[1], mMR_list[2], mMR_list[3] ))
            print("[MR_body, MR_part, mMR]: %.3f, %.3f, %.3f"%(MR_body, MR_part, mMR))
            error_list = [MR_body, MR_part, mMR]

        if data['dataset'] == "CrowdHuman":
            if len(json_dump) != 0 and len(json_dump_part_mr) != 0:
                AP_body, MR_body, AP_part, MR_part, mMR_list, mMR_avg = body_part_association_evaluation(
                    json_path, json_path_part_mr, data)
            else:
                AP_body, MR_body, AP_part, MR_part, mMR_avg = 0, 0, 0, 0, 0
                mMR_list = [0, 0, 0, 0]  # "Reasonable", "Small", "Heavy", "All"
            
            print("[AP@.5&MR]: AP_body: %.3f, AP_part: %.3f, MR_body: %.3f, MR_part: %.3f, mMR_avg: %.3f"%(
                AP_body, AP_part, MR_body, MR_part, mMR_avg ))
            print("[mMR_list]: Reasonable: %.3f, Small: %.3f, Heavy: %.3f, All: %.3f"%(
                mMR_list[0], mMR_list[1], mMR_list[2], mMR_list[3] ))
            error_list = [MR_body, MR_part, mMR_avg]
            # error_list = [MR_body, MR_part, mMR_list[-1]]  # All
            # error_list = [MR_body, MR_part, mMR_list[0]]  # Reasonable
            
        if data['dataset'] == "BodyHands":
            print("[BodyHands]: using <Cond. Accuracy> and <Joint AP> instead of <MR_body>, <MR_part> and <mMR> !")
            
            if len(json_dump) != 0:
                ap_dual, ap_single = body_part_association_evaluation(json_path, json_path_part_mr, data)
            else:
                ap_dual, ap_single = 0, 0
            
            print("AP_Dual(Joint-AP): %.3f, AP_Single: %.3f"%(ap_dual, ap_single))
            error_list = [1, 1, 1]
            
            
    if training:
        tmp.close()

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t2))  # speeds per image
    if not training and task != 'test':
        os.rename(json_path, osp.splitext(json_path)[0] + '_ap{:.4f}.json'.format(mAP))
        shape = (batch_size, 3, imgsz, imgsz)
        print(f'Speed: %.3fms pre-process, %.3fms inference, %.3fms NMS per image at shape {shape}' % t)

    model.float()  # for training
    # return (mp, mr, map50, mAP, *(loss.cpu() / len(dataloader)).tolist()), np.zeros(nc), t  # for compatibility with train
    return (mp, mr, map50, mAP, map50_part, mAP_part, *(loss.cpu() / len(dataloader)).tolist()), np.zeros(nc), t, error_list


def parse_opt():
    parser = argparse.ArgumentParser(prog='val.py')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', default='yolov5s6.pt')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--task', default='val', help='train, val, test')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--scales', type=float, nargs='+', default=[1])
    parser.add_argument('--flips', type=int, nargs='+', default=[-1])
    parser.add_argument('--rect', action='store_true', help='rectangular input image')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--pad', type=int, default=0, help='padding for two-stage inference')
    parser.add_argument('--json-name', type=str, default='', help='optional name for saved json file')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--augment', action='store_true', help='augmented inference')

    opt = parser.parse_args()
    opt.flips = [None if f == -1 else f for f in opt.flips]
    opt.data = check_file(opt.data)  # check file
    return opt


def main(opt):
    set_logging()
    print(colorstr('val: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    if opt.task in ('train', 'val', 'test'):  # run normally
        run(opt, **vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
