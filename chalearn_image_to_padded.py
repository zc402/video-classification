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

from config.defaults import get_override_cfg
from utils.chalearn import train_list, test_list, val_list

def pad_an_img(img_path:Path, target_path:Path):  # Pad and Overwrite
    target_path.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(img_path))
    h, w, c = img.shape
    new_img = np.zeros(shape=(h*2, w*2, c), dtype=img.dtype)
    new_img[h//2: h//2 + h, w//2: w//2 + w, :] = img
    cv2.imwrite(str(target_path), new_img)

def pad_images(label_list, img_root, pad_root):

    for (m,k,l) in tqdm(label_list):
        video = Path(img_root, m.replace('.avi', ''))  # originally M_xxxxx.avi, now a folder named M_xxxxx
        target_video = Path(pad_root, m.replace('.avi', ''))
        imgs = glob.glob(str(Path(video, '*.jpg')), recursive=False)
        for img in imgs:
            target_img = Path(target_video, Path(img).name)
            pad_an_img(img, target_img)

    # shutil.copytree(img_root, pad_root, ignore=shutil.ignore_patterns('K_*.avi'))  #性能太差
    # full_imgs = glob.glob(str(Path(pad_root, '**', '*.jpg')), recursive=True)
    # for img_path in tqdm(full_imgs):
    #     pad_an_img(img_path)


if __name__ == '__main__':
    cfg = get_override_cfg()

    img_root = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.IMG)
    pad_root = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.PAD)

    if pad_root.exists():
        print("Padding: pad folder already exist")
        exit()
    shutil.rmtree(pad_root, ignore_errors=True)

    pad_images(train_list, img_root, pad_root)
    pad_images(test_list, img_root, pad_root)
    pad_images(val_list, img_root, pad_root)