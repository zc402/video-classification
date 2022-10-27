import os
import glob
from pathlib import Path
import shutil
import sys
from tqdm import tqdm
import cv2
import numpy as np
from config.defaults import get_override_cfg
from multiprocessing import Pool

os.environ['MKL_THREADING_LAYER'] = 'GNU'

cfg = get_override_cfg()

densepose = Path(cfg.DENSEPOSE)
# sys.path.append(str(densepose))
apply_net = Path(densepose, 'apply_net.py')

# R_50_FPN_s1x
# yaml_path = Path(densepose, 'configs', 'densepose_rcnn_R_50_FPN_s1x.yaml')
# model_download = Path('pretrained', 'model_final_162be9.pkl').absolute()

# R_101_FPN_DL_s1x 	
yaml_path = Path(densepose, 'configs', 'densepose_rcnn_R_101_FPN_DL_s1x.yaml')
model_download = Path('pretrained', 'model_final_844d15.pkl').absolute()

pad_root = Path(cfg.CHALEARN.ROOT ,cfg.CHALEARN.PAD)
iuv_root = Path(cfg.CHALEARN.ROOT ,cfg.CHALEARN.IUV)

def to_iuv_one_folder(xxx_folder, name_of_set, ):
    xxx_name = Path(xxx_folder).name
    # avi_folders = glob.glob(str(Path(xxx_folder, "M_*")))
    # imgs = glob.glob(str(Path(xxx_folder, "M_*", "*.jpg")))
    jpg_wildcard = Path(xxx_folder, "M_*", "*.jpg")
    jpg_wildcard = str(jpg_wildcard)
    output_path = Path(iuv_root, name_of_set, xxx_name + '.pkl')
    if output_path.exists():
        print(f'ignore existed file: {str(output_path)}')
        return
    output_path = str(output_path)
    apply_net_command = f'python {str(apply_net)} dump {str(yaml_path)} \
        {model_download} \
        "{jpg_wildcard}" --output {output_path}'
    os.system(apply_net_command)

def to_iuv_one_folder_wrap(params):
    i, xxx_folder, name_of_set = params
    print(f'Processing {i}')
    to_iuv_one_folder(xxx_folder, name_of_set)

def to_iuv(cfg, name_of_set):

    Path(iuv_root).mkdir(exist_ok=True)

    # model_download = 'https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl'

    train_folder = Path(pad_root, name_of_set)  # 001 002 003 ...
    xxx_folders = glob.glob(str(Path(train_folder, "*")), recursive=False)
    xxx_folders = [d for d in xxx_folders if Path(d).is_dir()]
    param_list = []
    for i, xxx_folder in enumerate(xxx_folders):
        param_list.append((i, xxx_folder, name_of_set))

    if False:
        for param in tqdm(param_list):
            to_iuv_one_folder_wrap(param)
    else:
        pool = Pool(min(10, cfg.NUM_CPU))
        pool.map(to_iuv_one_folder_wrap, param_list)

to_iuv(cfg, 'train')
to_iuv(cfg, 'test')
to_iuv(cfg, 'valid')