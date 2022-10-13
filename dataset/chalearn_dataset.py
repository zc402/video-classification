from glob import glob
from pathlib import Path
from torch.utils.data import Dataset
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt 
from torchvision import transforms

from config.defaults import get_override_cfg
from utils.chalearn import get_labels, train_list, test_list

cfg = get_override_cfg()

# The crops and corresponding pixels


class ChalearnVideoDataset(Dataset):

    crop_resize = {'CropLHand': 40, 'CropRHand': 40, 'CropHead': 40, 'CropTorso': 40,
    'CropLArm': 100, 'CropRArm': 100, 'CropLHandArm': 100, 'CropRHandArm': 100, 'CropHeadTorso': 100,
    'CropBody': 200}

    def __init__(self, name_of_set:str) -> None:
        """name_of_set: train test val"""
        self.name_of_set = name_of_set

        # Load label list
        self.labels = get_labels(name_of_set)
        self.clip_len = cfg.CHALEARN.CLIP  # length of clip (frames)
    
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _pad_resize_img(self, img, new_size:int):  # Pad to square and resize
        h, w, c = img.shape
        m = max(h, w)
        nx = (m-w) // 2  # The x coord in new image
        ny = (m-h) // 2  # The y coord in new image
        new_img = np.zeros(shape=(m, m, c), dtype=img.dtype)
        new_img[ny:ny+h, nx:nx+w, :] = img  # A square image with original content at center
        resize_img = cv2.resize(new_img, (new_size, new_size), interpolation=cv2.INTER_CUBIC)
        return resize_img


    def _get_image_features(self, nsetx3x5img:Path):
        """
        nsetx3x5img: train/001/M_00068/00000.jpg
        """
        # size = 100  # pixels
        feature_folder_list = self.crop_resize.keys()
        res_dict = {key: None for key in feature_folder_list}
        for crop_folder_name in res_dict.keys():
            size = self.crop_resize[crop_folder_name]
            frame_path = Path(cfg.CHALEARN.ROOT, crop_folder_name, nsetx3x5img)
            if frame_path.exists():
                img = cv2.imread(str(frame_path))
                img = self._pad_resize_img(img, size)
            else:
                img = np.zeros((size, size, 3), dtype=np.uint8)
            input_tensor = self.preprocess(img)
            res_dict[crop_folder_name] = input_tensor

        return res_dict

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        m, k, l = label
        nsetx3x5 = Path(m).parent / Path(m).stem  # train/001/M_00068/
        avi_path = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.IMG, nsetx3x5)  # root/Images/train/001/M_00068/
        img_files = glob(str(avi_path / "*"))  # Images from 1 folder
        img_files = sorted(img_files)
        img_names = [Path(p).name for p in img_files]  # 00000.jpg 00005.jpg ...

        # Random / Uniform sampling
        # Random sampling
        possible_start_idx = len(img_names) - self.clip_len
        possible_start_idx = max(0, possible_start_idx)
        start_idx = random.randint(0, possible_start_idx)  # (randint: start/end both included)
        clip_indices = range(start_idx, start_idx + self.clip_len)
        clip_indices = [i % len(img_names) for i in clip_indices]  # If clip is larger than video length, then pick from start
        selected_imgs = [img_names[i] for i in clip_indices]

        nsetx3x5img_list = [Path(nsetx3x5, n) for n in selected_imgs]
        selected_features = [self._get_image_features(img) for img in nsetx3x5img_list]
        # Collect dicts
        collected_features = {}
        for key in selected_features[0].keys():
            features = [f[key] for f in selected_features]
            collected = np.stack(features)  # Stack time dim
            collected_features[key] = collected
        collected_features['label'] = l - 1  # Chalearn label starts from 1 while torch requires 0 
        return collected_features

def _test():
    dataset = ChalearnVideoDataset('train')
    counter = 0
    for batch in dataset:
        for i in range(0, 5):
            # cv2.imwrite(f'./debug/{counter}_{i}_L.jpg', batch['CropLHand'][i], )
            cv2.imwrite(f'./debug/{counter}_{i}_R.jpg', batch['CropRHand'][i], )
        counter = counter + 1
        # plt.imshow()
        # plt.show()
        # plt.imshow(batch['CropRHand'][4])
        # plt.show()
        # plt.cla()
        pass