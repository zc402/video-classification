import os
import glob
from pathlib import Path
import shutil
import sys
from tqdm import tqdm
import cv2
import numpy as np
from config.defaults import get_override_cfg

os.environ['MKL_THREADING_LAYER'] = 'GNU'

cfg = get_override_cfg()

def to_iuv(cfg, name_of_set='train'):

    pad_root = Path(cfg.CHALEARN.ROOT ,cfg.CHALEARN.PAD)
    iuv_root = Path(cfg.CHALEARN.ROOT ,cfg.CHALEARN.IUV)
    Path(iuv_root).mkdir(exist_ok=True)

    densepose = Path(cfg.DENSEPOSE)
    # sys.path.append(str(densepose))
    apply_net = Path(densepose, 'apply_net.py')
    yaml_path = Path(densepose, 'configs', 'densepose_rcnn_R_50_FPN_s1x.yaml')

    train_folder = Path(pad_root, name_of_set)  # 001 002 003 ...
    xxx_folders = glob.glob(str(Path(train_folder, "*")), recursive=False)
    xxx_folders = [d for d in xxx_folders if Path(d).is_dir()]
    for xxx_folder in tqdm(xxx_folders):
        xxx_name = Path(xxx_folder).name
        # avi_folders = glob.glob(str(Path(xxx_folder, "M_*")))
        # imgs = glob.glob(str(Path(xxx_folder, "M_*", "*.jpg")))
        jpg_wildcard = Path(xxx_folder, "**", "*.jpg")
        jpg_wildcard = str(jpg_wildcard)
        output_path = Path(iuv_root, name_of_set, xxx_name + '.pkl')
        output_path = str(output_path)
        apply_net_command = f'python {str(apply_net)} dump {str(yaml_path)} \
            https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl \
            "{jpg_wildcard}" --output {output_path}'
        os.system(apply_net_command)

to_iuv(cfg, 'train')
to_iuv(cfg, 'test')