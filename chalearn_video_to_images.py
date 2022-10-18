"""
Convert videos to jpgs
"""
import glob
import os
import cv2
from pathlib import Path
from tqdm import tqdm
import shutil

from config.defaults import get_override_cfg

def video2images(video: Path, img_folder: Path, interval):
    """
    Split the video into frame images
    :param video:
    :param img_folder: image folder of the video with same name.
    :return:
    """
    img_folder.mkdir(exist_ok=True)
    frame_num = 0
    cap = cv2.VideoCapture(str(video))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % interval == 0:
            img_path = img_folder / str(frame_num).zfill(5)
            img_path = img_path.with_suffix(".jpg")  # 00000.jpg
            cv2.imwrite(str(img_path), frame)
            assert img_path.is_file()  # check, because when failed, cv2.imwrite do not raise exception.
        frame_num = frame_num + 1


if __name__ == '__main__':
    cfg = get_override_cfg()
    sample_root = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.SAMPLE)
    img_root = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.IMG)
    interval = cfg.CHALEARN.IMG_SAMPLE_INTERVAL
    
    if img_root.exists():
        print("v to i: target exists")
        exit()
    # shutil.rmtree(img_root, ignore_errors=True)
    shutil.copytree(sample_root, img_root)  # To copy folder structure: ignore=shutil.ignore_patterns('*.avi')

    avi_list = glob.glob(str(Path(img_root, '**', '*.avi')), recursive=True)
    for video in tqdm(avi_list):
        video = Path(video)
        video2images(video, video.parent / video.stem, interval)  # 001/K_00001

    for video in avi_list:
        os.remove(video)
