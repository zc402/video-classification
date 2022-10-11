from pathlib import Path
import sys
import numpy as np
import pickle
from config.defaults import get_override_cfg
from tqdm import tqdm
import cv2
import glob 
from utils.chalearn import train_list, test_list

cfg = get_override_cfg()

densepose = Path(cfg.DENSEPOSE)
sys.path.append(str(densepose))

def load_iuv(pkl_path):
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)
    return result


def crop(img_path, target_path, bbox):
    if bbox is None:
        black = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(str(target_path), black)
        return
    x1, y1, x2, y2 = bbox
    img = cv2.imread(str(img_path))  # H W C
    cropped = img[y1:y2, x1:x2, :]
    cv2.imwrite(str(target_path), cropped)

def extract_crop(name_of_set):
    pad_root = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.PAD)
    iuv_root = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.IUV)
    crop_body_root = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.CROP_BODY)

    iuv_list = glob.glob(str(Path(iuv_root, name_of_set, "*.pkl")))
    for iuv in tqdm(iuv_list):
        iuv_res = load_iuv(iuv)
        
        for item in iuv_res:  # images
            # Path: ./train/001/M_00068/00000.jpg
            file_path = item['file_name']
            file_path = Path(file_path)
            x_img = file_path.name  # 00000.jpg
            x5 = file_path.parent.name  # M_00068
            x3 = Path(iuv).stem  # 001
            x3x5img = Path(x3, x5, x_img)  # 001/M_00068/00000.jpg
            pad_img_path = Path(pad_root, name_of_set, x3x5img)
            crop_img_path = Path(crop_body_root, name_of_set, x3x5img)
            crop_img_path.parent.mkdir(parents=True, exist_ok=True)

            if item['pred_boxes_XYXY'].size()[0] == 0:
                # Not detected
                crop(pad_img_path, crop_img_path, None)
                print(f"No box detection: {pad_img_path}")
            else:
                bbox = item['pred_boxes_XYXY'].cpu().numpy().astype(int)[0]  # shape: 4
                crop(pad_img_path, crop_img_path, bbox)
        

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