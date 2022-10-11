import os
import glob
from pathlib import Path
import shutil
from tqdm import tqdm
import cv2
import numpy as np

from config.defaults import get_override_cfg


cfg = get_override_cfg()

pad_root = cfg.CHALEARN.PAD_ROOT
iuv_root = cfg.CHALEARN.IUV_ROOT
Path(iuv_root).mkdir(exist_ok=True)

train_folder = Path(pad_root, 'train', 'train')  # 001 002 003 ...
xxx_folders = glob.glob(str(Path(train_folder, "*")), recursive=False)
xxx_folders = [d for d in xxx_folders if Path(d).is_dir()]
for xxx_folder in tqdm(xxx_folders):
    xxx_name = Path(xxx_folder).name
    # avi_folders = glob.glob(str(Path(xxx_folder, "M_*")))
    # imgs = glob.glob(str(Path(xxx_folder, "M_*", "*.jpg")))
    jpg_wildcard = Path(xxx_folder, "**", "M_*.jpg")
    jpg_wildcard = str(jpg_wildcard)
    output_path = Path(iuv_root, xxx_name + '.pkl')
    output_path = str(output_path)
    apply_net_command = f'python apply_net.py dump configs/densepose_rcnn_R_50_FPN_s1x.yaml \
        https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl \
        "{jpg_wildcard}" --output {output_path}.pkl'
    os.system(apply_net_command)