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
train_folder = Path(pad_root, 'train', 'train')  # 001 002 003 ...
xxx_folders = glob.glob(str(Path(train_folder, "*")), recursive=False)
xxx_folders = [d for d in xxx_folders if Path(d).is_dir()]
