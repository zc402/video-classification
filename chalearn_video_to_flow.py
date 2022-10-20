"""
Convert videos to flows
"""
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

def video2flow(video_relative_path, video_root, flow_root):
    # relative path: train/xxx/xxxxx.avi
    video_path = Path(video_root, video_relative_path)  
    flow_path = Path(flow_root, video_relative_path)  
    
    cap = cv2.VideoCapture(str(video_path))
    flow_list = []
    im1 = None
    im2 = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (60, 80))
        if im1 is None:
            im1 = frame  # Frist frame
            y = np.zeros((im1.shape[0], im1.shape[1], 2))
        else:
            im2 = frame
            y = flow(im1, im2)

            # Make space for the next frame
            im1 = im2
            im2 = None

        flow_list.append(y)
    flow_data = np.array(flow_list)
    # Range of optical flow: [-5, 5]
    # print(np.array(flow_list).max(), np.array(flow_list).min())
    # for f in flow_list:
    #     img = ((f[:, :, 0] + 5) / 10 *255).astype(np.uint8)
    #     cv2.imshow('win', img)
    #     cv2.waitKey(100)
    flow_folder = flow_path.parent
    flow_folder.mkdir(parents=True, exist_ok=True)
    flow_name = flow_path.stem + '.npy'
    new_flow_path = Path(flow_folder, flow_name)   #  /111111.npy
    np.save(new_flow_path, flow_data)


cfg = get_override_cfg()

sample_root = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.SAMPLE)  # Videos with class number sampled
flow_root = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.FLOW)

def v2f_wrapper(params):
    video_relative_path, sample_root, flow_root = params
    video2flow(video_relative_path, sample_root, flow_root)

param_list = []
avi_list = glob.glob(str(Path(sample_root, '**', '*.avi')), recursive=True)
for video in tqdm(avi_list):
    video = Path(video)
    name = video.name
    xxx = video.parent.name
    name_of_set = video.parent.parent.name
    video_relative_path = Path(name_of_set, xxx, name)
    param_list.append((video_relative_path, sample_root, flow_root))
    # video2flow(video_relative_path, sample_root, flow_root)
pool = Pool(16)
pool.map(v2f_wrapper, param_list)