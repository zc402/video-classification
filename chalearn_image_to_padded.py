"""
Pad videos to meet the human size of the dense pose
"""
import os
import glob
from pathlib import Path
import shutil
from tqdm import tqdm
import cv2
import numpy as np

from config.defaults import get_override_cfg

# def pad_single_video(
#     input_video:Path, 
#     output_video:Path):

#     assert input_video.is_file()
#     # Suggested by SlowFast(facebookresearch): ffmpeg -i "${video}" -r 30 -q:v 1 "${out_name}"
#     command = f'ffmpeg -i {input_video.absolute()} -vf "pad=iw*2:ih*2:iw/2:ih/2" {output_video.absolute()}'
#     execute_result = os.system(command)
#     if execute_result != 0:
#         print(execute_result)


# def pad_videos():
#     cfg = get_override_cfg()

#     root = cfg.CHALEARN.ROOT
#     root = Path(root)
#     pad_root = cfg.CHALEARN.PAD_ROOT
#     pad_root = Path(pad_root)

#     if pad_root.is_dir():
#         raise Exception(f'{pad_root} already exists')
    
#     shutil.copytree(root, pad_root)

#     avi_full = glob.glob(str(Path(pad_root, '**', '*.avi')), recursive=True)
#     for video in tqdm(avi_full):
#         video = Path(video)
#         pad_single_video(video, video.parent / ('pad_' + video.name))  # 001/pad_K_00001.avi

#     for video in avi_full:
#         os.remove(video)
    
#     avi_padded = glob.glob(str(Path(pad_root, '**', 'pad_*')), recursive=True)
#     for video in tqdm(avi_padded):
#         old_path = video
#         video = Path(video)
#         name = video.name
#         new_name = name.replace("pad_", "")
#         new_path = Path(video.parent, new_name)
#         os.rename(old_path, str(new_path))

def pad_an_img(img_path):  # Pad and Overwrite
    img = cv2.imread(str(img_path))
    h, w, c = img.shape
    new_img = np.zeros(shape=(h*2, w*2, c), dtype=img.dtype)
    new_img[h//2: h//2 + h, w//2: w//2 + w, :] = img
    cv2.imwrite(img_path, new_img)

def pad_images():
    cfg = get_override_cfg()

    img_root = cfg.CHALEARN.IMG_ROOT
    # img_root = '/home/zc/Downloads/denseposetest'
    img_root = Path(img_root)
    pad_root = cfg.CHALEARN.PAD_ROOT
    pad_root = Path(pad_root)

    if pad_root.is_dir():
        raise Exception(f'{pad_root} already exists')

    shutil.copytree(img_root, pad_root)
    full_imgs = glob.glob(str(Path(pad_root, '**', '*.jpg')), recursive=True)
    for img_path in tqdm(full_imgs):
        pad_an_img(img_path)


if __name__ == '__main__':
    pad_images()