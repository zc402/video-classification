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

cfg = get_override_cfg()

flow_root = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.FLOW)

npy_list = glob.glob(str(Path(flow_root, '**', '*.npy')), recursive=True)

for npy_path in npy_list:
    flow_data = np.load(npy_path)
    flowRGB_path = npy_path.replace(cfg.CHALEARN.FLOW, cfg.CHALEARN.FLOWRGB).replace('.npy', '.avi')
    
    # Range of optical flow: [-5, 5]
    # print(np.array(flow_list).max(), np.array(flow_list).min())
    for f in flow_data:
        img = ((f[:, :, 0] + 5) / 10 *255).astype(np.uint8)
        cv2.imshow('win', img)
        cv2.waitKey(100)