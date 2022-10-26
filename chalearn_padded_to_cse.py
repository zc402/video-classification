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

def to_CSE(cfg, name_of_set='train'):

    pad_root = Path(cfg.CHALEARN.ROOT ,cfg.CHALEARN.PAD)
    CSE_root = Path(cfg.CHALEARN.ROOT ,cfg.CHALEARN.CSE)
    Path(CSE_root).mkdir(exist_ok=True)

    densepose = Path(cfg.DENSEPOSE)
    # sys.path.append(str(densepose))
    apply_net = Path(densepose, 'apply_net.py')
    yaml_path = Path(densepose, 'configs', 'cse', 'densepose_rcnn_R_50_FPN_s1x.yaml')
    model_download = Path('pretrained', 'model_final_c4ea5f.pkl').absolute()
    # model_download = 'https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_s1x/251155172/model_final_c4ea5f.pkl'

    train_folder = Path(pad_root, name_of_set)  # 001 002 003 ...
    xxx_folders = glob.glob(str(Path(train_folder, "*")), recursive=False)
    xxx_folders = [d for d in xxx_folders if Path(d).is_dir()]
    for xxx_folder in tqdm(xxx_folders):
        xxx_name = Path(xxx_folder).name
        # avi_folders = glob.glob(str(Path(xxx_folder, "M_*")))
        # imgs = glob.glob(str(Path(xxx_folder, "M_*", "*.jpg")))
        jpg_wildcard = Path(xxx_folder, "**", "*.jpg")
        jpg_wildcard = str(jpg_wildcard)
        output_path = Path(CSE_root, name_of_set, xxx_name + '.pkl')
        if output_path.exists():
            print(f'ignore existed file: {str(output_path)}')
            continue
        output_path = str(output_path)
        apply_net_command = f'python {str(apply_net)} dump {str(yaml_path)} \
            {model_download} \
            "{jpg_wildcard}" --output {output_path}'
        os.system(apply_net_command)

to_CSE(cfg, 'train')
to_CSE(cfg, 'test')
to_CSE(cfg, 'valid')