from pathlib import Path
import sys
import numpy as np
import pickle
from config.defaults import get_override_cfg
from tqdm import tqdm
import cv2
import glob 
from utils.chalearn import train_list, test_list

import matplotlib.patches as patches
import matplotlib.pyplot as plt 

cfg = get_override_cfg()

densepose = Path(cfg.DENSEPOSE)
sys.path.append(str(densepose))

def load_iuv(pkl_path):
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)
    return result


def crop(img_path, target_path, bbox):
    if Path(target_path).exists():
        return  # Do not overwrite
    # if bbox is None:
    #     black = np.zeros((10, 10, 3), dtype=np.uint8)
    #     cv2.imwrite(str(target_path), black)
    #     return
    x1, y1, x2, y2 = bbox
    img = cv2.imread(str(img_path))  # H W C
    cropped = img[y1:y2, x1:x2, :]
    cv2.imwrite(str(target_path), cropped)


def crop_body_parts(human_img_path, target_relative_path, iuv):
    """
    human_img_path: path of cropped human image
    """

    lhand = 4
    rhand = 3
    I = iuv['pred_densepose'][0].labels  # pixel region segmentation results

    target_path = Path(cfg.CHALEARN.ROOT, 'CropLHand', target_relative_path)
    if Path(target_path).exists():
        return  # Do not overwrite
    target_path.parent.mkdir(parents=True, exist_ok=True)
    mask_lhand = (I == lhand)
    mask_lhand = mask_lhand.cpu().numpy().astype(np.uint8)
    contours, _ = cv2.findContours(mask_lhand,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)==0:
        return
    else:
        img = cv2.imread(str(human_img_path))
        if len(contours)==1:
            x, y, w, h = cv2.boundingRect(contours[0])
            if w < 15 or h < 15:
                return  # Too small, probably wrong
            cropped = img[y:y+h, x:x+w, :]
            # Show --------------------
            show = False
            if show:
                fig, ax = plt.subplots(1)
                ax.imshow(img)
                rect = patches.Rectangle((x, y), w, h, linewidth=1,
                                edgecolor='r', facecolor="none")
                ax.add_patch(rect)
                plt.show()
            
        else:  # len > 1
            # Show-------------
            show = False
            if show:
                fig, ax = plt.subplots(1)
                ax.imshow(img)
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    rect = patches.Rectangle((x, y), w, h, linewidth=1,
                                edgecolor='r', facecolor="none")
                    ax.add_patch(rect)
                plt.show()
                plt.close()
            # Write----------------
            area = []
            xywh = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area.append(w*h)
                xywh.append((x,y,w,h))
            amax = np.array(area).argmax()
            largest_xywh = xywh[amax]
            x,y,w,h = largest_xywh
            if w < 15 or h < 15:
                return  # Too small, probably wrong
            cropped = img[y:y+h, x:x+w, :]
            
        cv2.imwrite(str(target_path), cropped)


def extract_crop(name_of_set):
    pad_root = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.PAD)
    iuv_root = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.IUV)
    crop_body_root = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.CROP_BODY)

    iuv_list = glob.glob(str(Path(iuv_root, name_of_set, "*.pkl")))
    for iuv in tqdm(iuv_list):
        iuv_res = load_iuv(iuv)
        
        for iuv_item in iuv_res:  # images
            # Path: ./train/001/M_00068/00000.jpg
            file_path = iuv_item['file_name']
            file_path = Path(file_path)
            x_img = file_path.name  # 00000.jpg
            x5 = file_path.parent.name  # M_00068
            x3 = Path(iuv).stem  # 001
            x3x5img = Path(x3, x5, x_img)  # 001/M_00068/00000.jpg
            pad_img_path = Path(pad_root, name_of_set, x3x5img)
            crop_img_path = Path(crop_body_root, name_of_set, x3x5img)
            crop_img_path.parent.mkdir(parents=True, exist_ok=True)

            if iuv_item['pred_boxes_XYXY'].size()[0] == 0:
                # No detection
                # crop(pad_img_path, crop_img_path, None)
                print(f"No box detection: {pad_img_path}")
            else:
                bbox = iuv_item['pred_boxes_XYXY'].cpu().numpy().astype(int)[0]  # shape: 4
                crop(pad_img_path, crop_img_path, bbox)
                crop_body_parts(crop_img_path, x3x5img, iuv_item)
        

    # for (m,k,l) in tqdm(label_list):
    #     # k: train/002/M_00210.avi
    #     name_of_set = Path(m).parent.parent.name  # train
    #     x3 = Path(m).parent.name  # 001
    #     x5 = Path(m).stem  # M_00210, name of pad folder
    #     pad_video = Path(pad_root, name_of_set, x3, x5)
    #     iuv_path = Path(iuv_root, name_of_set, x3 + '.pkl')
    #     iuv_res = load_iuv(iuv_path)
    #     boxes = 
    #     pass


extract_crop('train')
extract_crop('test')