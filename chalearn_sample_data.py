import os
import glob
from pathlib import Path
import shutil
from tqdm import tqdm
import cv2
import numpy as np

from config.defaults import get_override_cfg

def cp(source:Path, dest:Path):
    if dest.exists():
        print("Sampling: video already exist")
        return
    dest.parent.parent.mkdir(exist_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, dest)

def sample_a_set(label_path:Path, video_folder:Path, new_root, allow_class):

    with label_path.open('r') as f:
        label_list = f.readlines()

    # Filter videos
    labels = [line.split(' ') for line in label_list]  # [M, K, L]
    labels = [(m, k, int(l)) for (m, k, l) in labels]
    labels = [(m, k, l) for (m, k, l) in labels if l <= allow_class]

    # Save new labels
    name_of_set = video_folder.name
    new_label_path = Path(new_root, name_of_set + ".txt")
    new_label_path.parent.mkdir(parents=True, exist_ok=True)
    label_str = [f'{m} {k} {l}\n' for (m, k, l) in labels]
    with new_label_path.open('w') as f:
        f.writelines(label_str)

    # copy videos to sample folder
    for label in tqdm(labels):
        m,k,l = label
        m_video = Path(video_folder, m)
        k_video = Path(video_folder, k)
        new_m_video = Path(new_root, m)
        new_k_video = Path(new_root, k)
        cp(m_video, new_m_video)
        cp(k_video, new_k_video)


cfg = get_override_cfg()

iso_root = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.ISO)
sample_root = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.SAMPLE)

sample_root.mkdir(exist_ok=True)
allow_class = cfg.CHALEARN.SAMPLE_CLASS

train_txt = Path(iso_root, 'IsoGD_labels', 'train.txt')
train_folder = Path(iso_root, 'train')
sample_a_set(train_txt, train_folder, sample_root, allow_class)

test_txt = Path(iso_root, 'IsoGD_labels', 'test.txt')
test_folder = Path(iso_root, 'test')
sample_a_set(test_txt, test_folder, sample_root, allow_class)

test_txt = Path(iso_root, 'IsoGD_labels', 'valid.txt')
test_folder = Path(iso_root, 'valid')
sample_a_set(test_txt, test_folder, sample_root, allow_class)