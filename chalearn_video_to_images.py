import glob
import os
import cv2
from pathlib import Path
from tqdm import tqdm

from config.defaults import get_cfg

def video2images(video: Path, img_folder: Path):
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
        img_path = img_folder / str(frame_num).zfill(5)
        img_path = img_path.with_suffix(".jpg")  # 00000.jpg
        cv2.imwrite(str(img_path), frame)
        assert img_path.is_file()  # when failed, cv2.imwrite do not raise exception.
        frame_num = frame_num + 1


if __name__ == '__main__':
    cfg = get_cfg()
    # video_root_path = "/home/zc/Datasets/[手势识别]ChaLearn/ChaLearn Isolated Gesture Recognition"
    img_root_path = "/home/zc/Datasets/ChaLearnIsoImages"
    avi_list = glob.glob(str(Path(img_root_path, '**', '*.avi')), recursive=True)
    for video in tqdm(avi_list):
        video = Path(video)
        video2images(video, video.parent / video.stem)

    for video in avi_list:
        os.remove(video)
