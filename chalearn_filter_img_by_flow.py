"""
Only keep images with enough flow energy
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
from numpy.linalg import norm 
import matplotlib.pyplot as plt


cfg = get_override_cfg()
flow_root = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.FLOW)
sample_video_root = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.SAMPLE)  # Videos with class number sampled
img_energy_root = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.IMG_ENERGY)


def video2images(video: Path):
    """
    Split the video into frame images
    :param video:
    :param img_folder: image folder of the video with same name.
    :return:
    """
    video = Path(video)
    v_stem = video.stem
    xxx = video.parent.name
    name_of_set = video.parent.parent.name

    flow_folder = Path(flow_root, name_of_set, xxx, v_stem)
    flow_files = glob.glob(str(Path(flow_folder, "*.jpg")))

    target_folder = Path(img_energy_root, name_of_set, xxx, v_stem)
    target_folder.mkdir(parents=True, exist_ok=True)

    keep = 0.3  # keep 50% images
    filtered_imgs = []
    materials = []
    for flow_file in flow_files:
        flow = cv2.imread(flow_file)

        magnitude = flow[2]
        energy = np.mean(magnitude)
        materials.append((flow_file, energy))
        # energy_list.append(energy)
        # if energy > 84.4:
        #     # Save image
        #     file_number = int(Path(flow_file).stem)
        #     filtered_imgs.append(file_number)
    materials.sort(key=lambda x: x[1])  
    num_keep = int(len(materials) * keep)
    num_keep = max(8, num_keep)  # At least keep 8 frames
    num_keep = min(len(materials), num_keep)  # No more than video length
    for _ in range(num_keep):
        filte_img_path = materials.pop()[0]  # pop last (largest energy)
        file_number = int(Path(filte_img_path).stem)
        filtered_imgs.append(file_number)

    if len(filtered_imgs) == 0:
        print(f"empty folder: {video}")

    frame_num = 0
    cap = cv2.VideoCapture(str(video))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num in filtered_imgs:
            img_name = str(frame_num).zfill(5) + ".jpg"
            img_path = Path(target_folder, img_name)
            cv2.imwrite(str(img_path), frame)

        frame_num = frame_num + 1

shutil.rmtree(img_energy_root, ignore_errors=True)
avi_list = glob.glob(str(Path(sample_video_root, '**', 'M_*.avi')), recursive=True)

params = []
for video in avi_list:
    params.append((video))

if cfg.DEBUG:
    for video in tqdm(avi_list):
        video2images(video)  # 001/K_00001
else:
    pool = Pool(cfg.NUM_CPU)
    pool.map(video2images, params)

pass