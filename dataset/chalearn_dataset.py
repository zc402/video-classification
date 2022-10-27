from glob import glob
from pathlib import Path
from torch.utils.data import Dataset
import cv2
import numpy as np
import random
import torch
import matplotlib.pyplot as plt 
from torchvision import transforms

from config import crop_cfg
from config.crop_cfg import crop_resize_dict, crop_folder_list
from utils.chalearn import get_labels, train_list, test_list
from torch.utils.data.dataloader import default_collate

# import line_profiler
# import atexit
# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)

# cfg = get_override_cfg()

# The crops and corresponding pixels


class ChalearnVideoDataset(Dataset):

    crop_resize = crop_resize_dict  # {"CropFolderName": size}

    def __init__(self, cfg, name_of_set:str) -> None:
        """name_of_set: train test val"""
        self.name_of_set = name_of_set
        self.cfg = cfg
        self.num_data_modality_channels = 9

        # Load label list
        self.labels = get_labels(name_of_set)
        self.clip_len = cfg.CHALEARN.CLIP_LEN  # length of clip (frames)

        # Preprocess for both train and test
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),  #  x/255, HWC -> CHW
            transforms.Normalize(
                mean=[0.45] * self.num_data_modality_channels, 
                std=[0.225] * self.num_data_modality_channels),
        ])



    def _pad_resize_img(self, img, new_size:int):  # Pad to square and resize
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        h, w, c = img.shape
        m = max(h, w)
        nx = (m-w) // 2  # The x coord in new image
        ny = (m-h) // 2  # The y coord in new image
        new_img = np.zeros(shape=(m, m, c), dtype=img.dtype)
        new_img[ny:ny+h, nx:nx+w, :] = img  # A square image with original content at center
        resize_img = cv2.resize(new_img, (new_size, new_size), interpolation=cv2.INTER_CUBIC)
        if len(resize_img.shape) == 2:
            resize_img = resize_img[:, :, np.newaxis]
        return resize_img

    def _augment(self, feature_dict) -> None:
        """
        Data augmentation for training
        """
        for (folder, size) in crop_resize_dict.items():
            if folder in feature_dict.keys():
                padding = size // 10

                augment = transforms.Compose([
                    transforms.RandomCrop(size, padding)
                ])
                feature_dict[folder] = augment(feature_dict[folder])


    def _get_image_features(self, nsetx3x5img:Path):
        """
        Get features (RGB UV ...) from image path
        nsetx3x5img: train/001/M_00068/00000.jpg
        """
        # size = 100  # pixels

        # res_dict = {key: None for key in crop_folder_list}
        res_dict = {self.cfg.MODEL.R3D_INPUT: None}  # Load less data to prevent memory fail
        for crop_folder_name in res_dict.keys():
            size = self.crop_resize[crop_folder_name]
            frame_path = Path(self.cfg.CHALEARN.ROOT, crop_folder_name, nsetx3x5img)
            if frame_path.exists():
                img = cv2.imread(str(frame_path))
                img_U = cv2.imread(str(Path(frame_path.parent, 'U_'+frame_path.name)), cv2.IMREAD_GRAYSCALE)
                img_V = cv2.imread(str(Path(frame_path.parent, 'V_'+frame_path.name)), cv2.IMREAD_GRAYSCALE)
                img_F = cv2.imread(str(Path(frame_path.parent, 'F_'+frame_path.name)))
                img_D = cv2.imread(str(Path(frame_path.parent, 'D_'+frame_path.name)), cv2.IMREAD_GRAYSCALE)
                img, img_U, img_V, img_F, img_D = [self._pad_resize_img(x, size) for x in (img, img_U, img_V, img_F, img_D)]  # HWC
                img_mul = np.concatenate([img, img_U, img_V, img_F, img_D], axis=-1)
            else:
                img_mul = np.zeros((size, size, self.num_data_modality_channels), dtype=np.uint8) + 127
            input_tensor = self.preprocess(img_mul)
            res_dict[crop_folder_name] = input_tensor

        return res_dict

    def random_sampling(self, seq_len, clip_len):
        possible_start_idx = seq_len - clip_len
        possible_start_idx = max(0, possible_start_idx)
        start_idx = random.randint(0, possible_start_idx)  # (randint: start/end both included)
        clip_indices = range(start_idx, start_idx + clip_len)
        clip_indices = [i % seq_len for i in clip_indices]  # If clip is larger than video length, then pick from start
        return clip_indices
    
    def uniform_sampling(self, seq_len, clip_len):
        clips = []
        if(seq_len <= clip_len):
            clips.append(self.random_sampling(seq_len, clip_len))
        else:
            t = 0
            for t in range(0, seq_len - clip_len, 4):
                clip_indices = range(t, t + clip_len)
                clips.append(clip_indices)
        return clips

    def collect_features_from_indices(self, clip_indices, img_names, img_folder, label):
        """Collect features from indices like [5, 10, 15, ...]"""
        selected_imgs = [img_names[i] for i in clip_indices]
        nsetx3x5img_list = [Path(img_folder, n) for n in selected_imgs]
        selected_features = [self._get_image_features(img) for img in nsetx3x5img_list]

        # Collect dicts
        collected_features = {}  # TCHW
        for key in selected_features[0].keys():
            features = [f[key] for f in selected_features]
            collected = torch.stack(features)  # Stack time dim
            collected_features[key] = collected
        collected_features['label'] = label - 1  # Chalearn label starts from 1 while torch requires 0 
        if self.name_of_set == "train":
            self._augment(collected_features)
        return collected_features

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        m, k, l = label
        nsetx3x5 = Path(m).parent / Path(m).stem  # train/001/M_00068/
        avi_path = Path(self.cfg.CHALEARN.ROOT, self.cfg.CHALEARN.IMG, nsetx3x5)  # root/Images/train/001/M_00068/
        img_files = glob(str(avi_path / "*"))  # Images from 1 folder
        img_files = sorted(img_files)
        img_names = [Path(p).name for p in img_files]  # 00000.jpg 00005.jpg ...

        # Random / Uniform sampling
        # Random sampling
        seq_len = len(img_names)
        if self.name_of_set == "train" or self.name_of_set == "valid":
            # Random sampling, take 1
            clip_indices = self.random_sampling(seq_len, self.clip_len)
            collected_features = self.collect_features_from_indices(clip_indices, img_names, nsetx3x5, l)
        else:
            # Uniform sampling, take multiple
            clip_indices_list = self.uniform_sampling(seq_len, self.clip_len)
            collected_features = [self.collect_features_from_indices(x, img_names, nsetx3x5, l) for x in clip_indices_list]
            # collected_features = default_collate(collected_features)
        
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