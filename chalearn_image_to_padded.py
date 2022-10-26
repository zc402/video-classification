"""
Pad videos to meet the human size of the dense pose
"""
import os
import glob
from pathlib import Path
import shutil
from tqdm import tqdm
import cv2
import numpy as np
from multiprocessing import Pool

from config.defaults import get_override_cfg
from utils.chalearn import train_list, test_list, val_list

def pad_an_img(img_path:Path, target_path:Path):  # Pad and Overwrite
    target_path.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(img_path))
    h, w, c = img.shape
    new_img = np.zeros(shape=(h*2, w*2, c), dtype=img.dtype)
    new_img[h//2: h//2 + h, w//2: w//2 + w, :] = img
    cv2.imwrite(str(target_path), new_img)

def pad_images_loop_wrapper(x_list):
    pad_images_loop(*x_list)

def pad_images(label_list, img_root, pad_root):

    pool = Pool(16)
    
    param_list = []
    for mkl in label_list:
        param_list.append((mkl, img_root, pad_root))
    
    pool.map(pad_images_loop_wrapper, param_list)

def pad_images_loop(mkl, img_root, pad_root):
    m,k,l = mkl
    for modality in (m, k):  # m: RGB, k: depth
        M_xxxxx = modality.replace('.avi', '')
        video = Path(img_root, M_xxxxx)  # originally M_xxxxx.avi, now a folder named M_xxxxx
        target_video = Path(pad_root, M_xxxxx)
        imgs = glob.glob(str(Path(video, '*.jpg')), recursive=False)
        for img in imgs:
            target_img = Path(target_video, Path(img).name)
            pad_an_img(img, target_img)

if __name__ == '__main__':
    cfg = get_override_cfg()

    img_root = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.IMG)
    pad_root = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.PAD)

    # if pad_root.exists():
    #     print("Padding: pad folder already exist")
    #     exit()
    shutil.rmtree(pad_root, ignore_errors=True)

    pad_images(train_list, img_root, pad_root)
    pad_images(test_list, img_root, pad_root)
    pad_images(val_list, img_root, pad_root)