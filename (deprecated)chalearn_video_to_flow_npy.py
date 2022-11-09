"""
Convert videos to flows, full resolution, save as numpy
"""
from fileinput import filename
import glob
import os
import cv2
from pathlib import Path
from tqdm import tqdm
import shutil
from pyflow import pyflow
from config.defaults import get_override_cfg
import numpy as np
from multiprocessing import Pool
import sys
from numpy.linalg import norm 
import io



def flow(im1, im2):

    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.

    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    u, v, im2W = pyflow.coarse2fine_flow(
        im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)

    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    return flow

def video2flow(video_relative_path, video_root, flow_npy_root):
    # relative path: train/xxx/xxxxx.avi
    video_path = Path(video_root, video_relative_path)  
    flow_npy_path = Path(flow_npy_root, video_relative_path)
    flow_npy_path = Path(str(flow_npy_path).replace('.avi', ''))
    if flow_npy_path.exists():
        return
    flow_npy_path.parent.mkdir(exist_ok=True, parents=True)

    cap = cv2.VideoCapture(str(video_path))
    flow_list = []
    im1 = None
    im2 = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        assert im2 is None
        if im1 is None:  # Frist frame
            im1 = frame
            im2 = frame
        else:
            im2 = frame

        y = flow(im1, im2)

        # Make space for the next frame
        im1 = im2
        im2 = None

        flow_list.append(y)
    
    np.save(str(flow_npy_path), np.array(flow_list))

cfg = get_override_cfg()

sample_root = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.SAMPLE)  # Videos with class number sampled
flow_root = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.FLOW_NPY)

def v2f_wrapper(params):
    video_relative_path, sample_root, flow_root = params
    video2flow(video_relative_path, sample_root, flow_root)

param_list = []
avi_list = glob.glob(str(Path(sample_root, '**', 'M_*.avi')), recursive=True)
for video in avi_list:
    video = Path(video)
    name = video.name
    xxx = video.parent.name
    name_of_set = video.parent.parent.name
    video_relative_path = Path(name_of_set, xxx, name)
    param_list.append((video_relative_path, sample_root, flow_root))

if cfg.DEBUG:
    for p in tqdm(param_list):
        v2f_wrapper(p)
else:
    pool = Pool(cfg.NUM_CPU)
    pool.map(v2f_wrapper, param_list)
