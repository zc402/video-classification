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
from utils.chalearn import train_list, test_list

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

def pad_an_img(img_path:Path, target_path:Path):  # Pad and Overwrite
    target_path.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(img_path))
    h, w, c = img.shape
    new_img = np.zeros(shape=(h*2, w*2, c), dtype=img.dtype)
    new_img[h//2: h//2 + h, w//2: w//2 + w, :] = img
    cv2.imwrite(str(target_path), new_img)

def pad_images(label_list, img_root, pad_root):


    for (m,k,l) in tqdm(label_list):
        video = Path(img_root, m.replace('.avi', ''))  # originally M_xxxxx.avi, now a folder named M_xxxxx
        target_video = Path(pad_root, m.replace('.avi', ''))
        imgs = glob.glob(str(Path(video, '*.jpg')), recursive=False)
        for img in imgs:
            target_img = Path(target_video, Path(img).name)
            pad_an_img(img, target_img)

    # shutil.copytree(img_root, pad_root, ignore=shutil.ignore_patterns('K_*.avi'))  #性能太差
    # full_imgs = glob.glob(str(Path(pad_root, '**', '*.jpg')), recursive=True)
    # for img_path in tqdm(full_imgs):
    #     pad_an_img(img_path)


if __name__ == '__main__':
    cfg = get_override_cfg()

    img_root = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.IMG)
    pad_root = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.PAD)

    shutil.rmtree(pad_root, ignore_errors=True)

    pad_images(train_list, img_root, pad_root)
    pad_images(test_list, img_root, pad_root)