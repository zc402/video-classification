import glob
import os
import cv2
from pathlib import Path
from tqdm import tqdm
import shutil

from config.defaults import get_override_cfg
import numpy as np
from multiprocessing import Pool
import sys
from numpy.linalg import norm 

cfg = get_override_cfg()

flow_root = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.FLOW)

npy_list = glob.glob(str(Path(flow_root, '**', '*.npy')), recursive=True)

for npy_path in npy_list:
    flow_data = np.load(npy_path)
    flowRGB_path = npy_path.replace(cfg.CHALEARN.FLOW, cfg.CHALEARN.FLOWRGB).replace('.npy', '.avi')
    
    # Range of optical flow: [-5, 5]
    # print(np.array(flow_list).max(), np.array(flow_list).min())
    for f in flow_data:
        # [-5, 5] -> [0, 1]
        f01 = (f + 5) / 10
        M_norm = norm(f01, ord=2, axis=2)  # HWC
        rgb = np.concatenate([f01, M_norm[:, :, np.newaxis]], axis=2)
        # img = ( *255).astype(np.uint8)
        cv2.imshow('win', cv2.resize(rgb, dsize=None, fx=4, fy=4))
        cv2.waitKey(100)