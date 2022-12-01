import glob
import pickle
import sys
from typing import List, Optional
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision
from tqdm import tqdm

from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

import tkinter  # For matplotlib
import matplotlib

import matplotlib.pyplot as plt

import torchvision.transforms.functional as F
import av



class ChaPath:
    def __init__(self, path) -> None:
        # /media/zc/C2000Pro-1TB/ChaLearnIsoNew/1_Sample/train/001/K_00002.avi
        self.path = Path(path)
        pass

    def change_split(self, name_of_set):
        # train

        assert self.path.parts[-3] in ['train', 'valid', 'test']
        assert name_of_set in ['train', 'valid', 'test']
        parts = list(self.path.parts)
        parts[-3] = name_of_set
        new_path = Path(*parts)
        return new_path

    def change_base(self, base):
        # 3_Pad
        
        parts = list(self.path.parts)
        parts[-4] = base
        new_path = Path(*parts)
        return new_path
    
    def prepend(self, s):
        parts = list(self.path.parts)
        parts[-1] = s + parts[-1]
        new_path = Path(*parts)
        return new_path
    

# chaPath = ChaPath('/media/zc/C2000Pro-1TB/ChaLearnIsoNew/1_Sample/train/001/M_00084.avi')
from PIL import Image
# from pims import PyAVReaderIndexed
import decord
from decord import VideoReader
decord.bridge.set_bridge('torch')
class VideoIO:
    
    @staticmethod
    def write_video(filename: Path, video_array: np.ndarray):
        assert len(video_array.shape) == 3  # (T,H,W), gray
        T, H, W = video_array.shape

        filename.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(filename), mode="w") as container:

            stream = container.add_stream("mpeg4", rate=10)
            stream.width = W
            stream.height = H
            stream.pix_fmt = "yuv420p"

            for frame in video_array:

                frame = av.VideoFrame.from_ndarray(frame, format="gray")
                for packet in stream.encode(frame):
                    container.mux(packet)

            for packet in stream.encode():
                container.mux(packet)
    
    @staticmethod
    def write_video_TCHW(filename: Path, video_array: np.ndarray):
        assert len(video_array.shape) == 4  #  TCHW
        T, C, H, W = video_array.shape
        for c in range(C):
            cPath = Path(filename.parent, f'{c}_{filename.name}')
            # save_path = ChaPath(filename).prepend(f'{c}_')
            VideoIO.write_video(cPath, video_array[:, c])

    @staticmethod
    def read_video_TCHW(
        filename: Path, 
        channels: int, 
        frames: List[int], 
        # xyxy: Optional[List[int]], 
        format: str = 'gray'):
        assert format in ['rgb24', 'gray']

        # def crop_video(video, clip_frames, clip_box):
        #     if clip_box is None:
        #         frames = video[clip_frames]
        #     else:
        #         assert len(clip_box) == 4
        #         x1, y1, x2, y2 = clip_box
        #         frames = video[clip_frames][y1: y2, x1: x2]
        #     return frames

        if format == 'rgb24':
            # Read rgb as shape TCHW
            assert filename.is_file()
            # video = PyAVReaderIndexed(filename.as_posix())  # THWC  # TODO: fromat
            vr = VideoReader(filename.as_posix())  # Color frames with channels = 3
            video = vr.get_batch(frames)  # THWC
            video = torch.permute(video, (0, 3, 1, 2))  # TCHW
            return video

        elif format == 'gray':
            videos = []
            for c in range(channels):
                cPath = Path(filename.parent, f'{c}_{filename.name}')
                assert cPath.is_file()
                # video = PyAVReaderIndexed(cPath.as_posix())  # THWC  

                vr = VideoReader(cPath.as_posix())
                video = vr.get_batch(frames)  # THWC
                video_gray = video[:, :, :, 0]  # THW
                videos.append(video_gray)
            videos = torch.stack(videos, dim=1)  # TCHW
            return videos

    # @staticmethod
    # def read_video_TCHW(filename: Path, channels: int):
    #     videos = []
    #     for c in range(channels):
    #         cPath = Path(filename.parent, f'{c}_{filename.name}')
    #         frames = VideoIO.read_video_gray(cPath)
    #         videos.append(frames)  # THW
    #     videos = torch.stack(videos, dim=1)  # TCHW

    #     return videos
    
    @staticmethod
    def read_video(filename: Path, format):
        assert format in ['gray', 'rgb24']
        frame_list = []
        with av.open(filename.as_posix()) as container:
            for frame in container.decode(video=0):
                frame_list.append(frame.to_ndarray(format=format))

        # video_array = np.stack(frame_list)
        return frame_list


class ConvertVideoToFlow:
    def __init__(self, cfg) -> None:
        
        self.video_sample_root = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.SAMPLE)
        self.flow_base = cfg.CHALEARN.FLOW_VIDEO
        self.batch_size = 100

        self.model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=True).cuda()
        self.model.eval()

    def convert(self):

        avi_list = glob.glob(str(Path(self.video_sample_root, '**', 'M_*.avi')), recursive=True)
        for avi in tqdm(avi_list):
            flow_arr = self._flow_from_file(avi)  # T 2 H W
            if flow_arr is None:
                print(f'Skip corrupted file at {avi}')
                continue
            flow_file = ChaPath(avi).change_base(self.flow_base)
            VideoIO.write_video_TCHW(flow_file, flow_arr)
            # flow_file_0 = ChaPath(flow_file).prepend('F0_')
            # flow_file_1 = ChaPath(flow_file).prepend('F1_')
            # VideoIO.write_video(flow_file_0, flow_arr[:, 0])
            # VideoIO.write_video(flow_file_1, flow_arr[:, 1])

    def _flow_from_file(self, filename:Path):

        frames = VideoIO.read_video(filename, format='rgb24')
        T = len(frames)
        
        flow_list = []
        img1_batch = []
        img2_batch = []
        for t in range(T-1):
            img1_batch.append(frames[t])
            img2_batch.append(frames[t+1])

            if len(img1_batch) >= self.batch_size:
                flow = self._array_to_flow(np.asarray(img1_batch), np.asarray(img2_batch))
                flow_list.append(flow)
                img1_batch = []
                img2_batch = []
        
        if len(img1_batch) > 0:
            flow = self._array_to_flow(np.asarray(img1_batch), np.asarray(img2_batch))
            flow_list.append(flow)
        
        if len(flow_list) == 0:
            return None
        flow_arr = np.concatenate(flow_list, axis=0)  # T, 2, H, W
        return flow_arr

    def _array_to_flow(self, img1_batch, img2_batch):
        weights = Raft_Large_Weights.DEFAULT
        transforms = weights.transforms()
        img1_batch = torch.from_numpy(img1_batch).permute((0, 3, 1, 2))  # NHWC -> NCHW
        img2_batch = torch.from_numpy(img2_batch).permute((0, 3, 1, 2))
        img1_batch, img2_batch = transforms(img1_batch, img2_batch)

        with torch.no_grad():
            list_of_flows = self.model(img1_batch.cuda(), img2_batch.cuda())
        flow = list_of_flows[-1]
        flow = flow.cpu().numpy()

        flow = np.clip(flow, -30, 30) / 60 + 0.5  # 0,1
        flow = np.clip(flow * 255, 0, 255)
        flow = flow.astype(np.uint8)  # TCHW
        return flow

# ConvertVideoToFlow().convert()

class ConvertVideoToIUVPkl:
    def __init__(self, cfg) -> None:
        self.iuv_base = cfg.CHALEARN.IUV_NEW
        
        self.video_sample_root = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.SAMPLE)
        densepose = Path(cfg.DENSEPOSE)

        sys.path.insert(0, str(densepose))
        import apply_net
        self.apply_net = apply_net
        self.yaml_path = Path(densepose, 'configs', 'densepose_rcnn_R_101_FPN_DL_s1x.yaml').as_posix()
        self.model_pkl = Path('pretrained', 'model_final_844d15.pkl').absolute().as_posix()

    def convert(self):

        avi_list = glob.glob(str(Path(self.video_sample_root, '**', 'M_*.avi')), recursive=True)
        for avi in tqdm(avi_list):
            iuv_pkl_path = ChaPath(avi).change_base(self.iuv_base)
            iuv_pkl_path = iuv_pkl_path.with_suffix('.pkl')
            if iuv_pkl_path.exists():
                continue
            iuv_pkl_path.parent.mkdir(parents=True, exist_ok=True)

            self.apply_net.entrance(
                self.yaml_path, 
                self.model_pkl,
                avi,
                iuv_pkl_path.as_posix())


class ConvertIuvPklToUvVideo:
    """
    Convert the UV maps in .pkl file into individual videos
    """
    def __init__(self, cfg) -> None:
        self.iuv_base = cfg.CHALEARN.IUV_NEW
        self.uv_vid_base = cfg.CHALEARN.UV_VIDEO

        self.y_pad = 120  # y-pad-left
        self.x_pad = 160  # x-pad-left
        
        self.img_h = 240
        self.img_w = 320

        self.img_pad_h = 240*2
        self.img_pad_w = 320*2
        
        densepose = Path(cfg.DENSEPOSE).absolute()
        sys.path.append(str(densepose))
        
        self.iuv_pkl_folder = Path(cfg.CHALEARN.ROOT, self.iuv_base)
        self.pkl_list = glob.glob(str(Path(self.iuv_pkl_folder, '**', '*.pkl')), recursive=True)

    
    def save_uv(self, iuv_pkl: Path, save_path:Path):
        with open(iuv_pkl, 'rb') as f:
            results = pickle.load(f)

        uv_map_list = []  # [T][CHW]
        for result in results:  # one result is one frame
            bg_pad = np.zeros((2, self.img_pad_h, self.img_pad_w), np.uint8)  # CHW
            box = result['pred_boxes_XYXY']
            if len(box) == 0:
                # No detection
                print('No detection')
            else:
                x1, y1, x2, y2 = box[0].cpu().numpy().astype(int)
                
                densepose = result['pred_densepose'][0].uv  # CHW
                densepose = densepose.cpu().numpy()
                densepose = densepose * 255.
                densepose = densepose.astype(np.uint8)
                
                map_h, map_w = densepose.shape[1:]
                bg_pad[:, y1:y1+map_h, x1:x1+map_w] = densepose

            uv_map = bg_pad[  # Global pad to global un-pad
                :, 
                self.y_pad: self.y_pad + self.img_h, 
                self.x_pad: self.x_pad + self.img_w]

            uv_map_list.append(uv_map)
        uv_map_arr = np.stack(uv_map_list)
        VideoIO.write_video_TCHW(save_path, uv_map_arr)

    def save_by_pkl(self, pkl_path):
        uv_vid_path = ChaPath(pkl_path).change_base(self.uv_vid_base)
        uv_vid_path = uv_vid_path.with_suffix('.avi')
        if ChaPath(uv_vid_path).prepend('0_').exists():
            return

        uv_vid_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_uv(pkl_path, uv_vid_path)
    
    def convert(self):
        for pkl_path in tqdm(self.pkl_list):
            self.save_by_pkl(pkl_path)
    
    def convert_multithread(self):
        # Multi thread with tqdm shown
        # Performance dropped due to random access into machanical hard drive.
        pkl_list = self.pkl_list
        save_by_pkl = self.save_by_pkl
        from torch.utils.data import Dataset, DataLoader
        class MTDataset(Dataset):
            def __len__(self):
                return len(pkl_list)

            def __getitem__(self, index) -> None:
                save_by_pkl(pkl_list[index])
                return None

        dataset = MTDataset()
        loader = DataLoader(dataset, batch_size=100, num_workers=10, collate_fn=lambda x:x, drop_last=False)
        [_ for _ in tqdm(loader)]

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
class ConvertIuvPklToPartBox:
    """
    Extract boxes for parts from Iuv .pkl file
    """
    def __init__(self, cfg) -> None:
        self.iuv_base = cfg.CHALEARN.IUV_NEW

        self.y_pad = 120  # y-pad-left
        self.x_pad = 160  # x-pad-left
        
        self.img_h = 240
        self.img_w = 320
        
        self.num_parts = 25  # 0~24, 0 is background

        self.box_base = cfg.CHALEARN.BOX
        self.video_base = cfg.CHALEARN.SAMPLE

        densepose = Path(cfg.DENSEPOSE).absolute()
        sys.path.append(str(densepose))

        self.iuv_pkl_folder = Path(cfg.CHALEARN.ROOT, self.iuv_base)

    def get_box_from_part(self, label_map:np.ndarray, part_idx):
        """
        Return the BODY LOCAL coordinate of the biggest bounding box of a part (index of surface)
        Format: XYXY
        """

        part_mask = (label_map == part_idx).astype(np.uint8)
        contours, _ = cv2.findContours(part_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)==0:
            # Part not detected
            return None

        # Find biggest countour
        area = []
        xywh = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area.append(w*h)
            xywh.append((x,y,w,h))
        amax = np.array(area).argmax()
        largest_xywh = xywh[amax]
        x,y,w,h = largest_xywh
        if w < 15 or h < 15:
            return None  # Discard abnormally small parts (probably detected wrongly)
        
        return (x, y, x+w, y+h)

    def save_box(self, iuv_pkl:Path, box_path:Path):
        with open(iuv_pkl, 'rb') as f:
            results = pickle.load(f)
        
        box_list = []  # [T][Box, xyxy]
        for result in results:
            box_part = [None for _ in range(self.num_parts)]
            human_box = result['pred_boxes_XYXY']
            if len(human_box) == 0:
                # No detection
                pass
                # print('No detection')
            else:
                hx1, hy1, hx2, hy2 = human_box[0].cpu().numpy().astype(int)
                labels = result['pred_densepose'][0].labels  # The I in IUV
                labels = labels.cpu().numpy()
                for p in range(1, self.num_parts):
                    xyxy = self.get_box_from_part(labels, p)
                    if xyxy is not None:
                        x1, y1, x2, y2 = xyxy
                        # Body coordinate to global coordinate (original image, no pad)
                        x1, x2 = np.array([x1, x2]) + hx1 - self.x_pad
                        y1, y2 = np.array([y1, y2]) + hy1 - self.y_pad

                        xyxy = (x1, y1, x2, y2)
                    box_part[p] = xyxy
            box_list.append(box_part)

        with box_path.open('wb') as f:
            pickle.dump(box_list, f)
                

    def convert(self):
        pkl_list = glob.glob(str(Path(self.iuv_pkl_folder, '**', '*.pkl')), recursive=True)
        for pkl_path in tqdm(pkl_list):
            box_path = ChaPath(pkl_path).change_base(self.box_base)

            box_path.parent.mkdir(parents=True, exist_ok=True)
            self.save_box(pkl_path, box_path)

    def plot(self):
        pkl_list = glob.glob(str(Path(self.iuv_pkl_folder, '**', '*.pkl')), recursive=True)
        pkl = pkl_list[10]
        box_path = ChaPath(pkl).change_base(self.box_base)
        video_path = ChaPath(pkl).change_base(self.video_base)
        video_path = video_path.with_suffix('.avi')

        video = VideoIO.read_video(video_path, format='rgb24')
        with open(box_path, 'rb') as f:
            boxes_T = pickle.load(f)

        plt.ion()
        fig, ax = plt.subplots(1)
        ax.xaxis.tick_top()
        ax.yaxis.tick_left() 
        for frame, boxes in zip(video, boxes_T):

            ax.imshow(frame)
            for box in boxes:
                if box is None:
                    continue
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2,
                            edgecolor='r', facecolor="none")
                ax.add_patch(rect)
            
            plt.draw()
            plt.pause(0.1)
            plt.gca().cla()
            


class PartCompose:

    # Individual parts (surface composition)

    lHand = [4]
    rHand = [3]

    lUpArm = [15, 17]
    rUpArm = [16, 18]

    lLoArm = [19, 21]  # Left lower arm
    rLoArm = [20, 22]

    torso = [1, 2]
    head = [23, 24]

    # Part compositions

    lArm = lUpArm + lLoArm
    rArm = rUpArm + rLoArm

    TorsoArmHand = torso + lArm + rArm + lHand + rHand

    lHandLoArm = lHand + lLoArm
    lHandArm = lHand + lArm
    lHandArmTorso = lHand + lArm + torso

    rHandLoArm = rHand + rLoArm
    rHandArm = rHand + rArm
    rHandArmTorso = rHand + rArm + torso

    def combine_box_xyxy(self, box_arr:np.ndarray):
        """
        Args:
            box_arr: shape (N,4)
        """
        assert len(box_arr) > 0
        box_arr = np.array(box_arr)
        
        x1 = box_arr[:, 0]
        y1 = box_arr[:, 1]
        x2 = box_arr[:, 2]
        y2 = box_arr[:, 3]

        x1_min = min(x1)
        y1_min = min(y1)
        x2_max = max(x2)
        y2_max = max(y2)
        large_box = (x1_min, y1_min, x2_max, y2_max)
        return large_box

    def combine_spatial_box_xyxy(self, part_boxes, part_list):
        """
        Input boxes shape (P, 4), return combined boxes, shape (4)
        Args:
            part_boxes: list of boxes, shape [P][4], P = num parts
            part_list: list of combined parts, shape [P]
        """
        boxes = [part_boxes[p] for p in part_list]
        boxes = [b for b in boxes if b is not None]
        if len(boxes) == 0:
            return None
        else:
            box_arr = np.array(boxes)  # P, 4
            large_box = self.combine_box_xyxy(box_arr)
            return large_box
    
    def combine_temporal_box_xyxy(self, temporal_part_boxes, part_list):
        """
        Input boxes shape (T, P, 4), return combined boxes shape (4)
        Args:
            temporal_part_boxes: list of boxes, shape [T, P, 4]
        """
        box_TPB = [self.combine_spatial_box_xyxy(part_boxes, part_list) for part_boxes in temporal_part_boxes]
        box_TB = [x for x in box_TPB if x is not None]

        box_B = self.combine_box_xyxy(box_TB)
        return box_B

    def __init__(self) -> None:
        pass

from utils.chalearn import Labels

import random

class ChalearnGestureDataset(Dataset):

    def __init__(self, cfg, name_of_set:str, parts:List, sampling:str, do_augment:bool=False) -> None:
        """
        Args
            cfg: config node
            name_of_set: train, test, valid
            parts: List of parts (surfaces) which are combined later in the dataset.
        """
        super().__init__()
        assert name_of_set in ['train', 'test', 'valid']
        assert sampling in ['random', 'uniform']

        self.label_list = Labels(cfg).from_set(name_of_set)  # (M:rgb, K:depth, L)
        self.parts = parts  # e.g. compose.TorsoArmHand
        self.clip_len = cfg.CHALEARN.CLIP_LEN
        
        self.root = cfg.CHALEARN.ROOT
        self.sample_base = cfg.CHALEARN.SAMPLE
        self.box_base = cfg.CHALEARN.BOX
        self.flow_base = cfg.CHALEARN.FLOW_VIDEO
        self.uv_base = cfg.CHALEARN.UV_VIDEO
        self.input_size = cfg.MODEL.INPUT_SIZE

        self.sampling = sampling
        self.do_augment = do_augment  # Optional augment: crop&pad
        self.compose = PartCompose()

        self.trans_resize = torchvision.transforms.Resize(size=(self.input_size, self.input_size))
        self.trans_crop = torchvision.transforms.RandomCrop(size=self.input_size, padding=self.input_size // 10)
    
    def __len__(self):
        return len(self.label_list)

    def _preprocess(self, tensors:List[torch.Tensor]):
        """
        Args:
            tensors: List of tensors shape [TCHW]
        """
        num_cs = [x.size()[1] for x in tensors]
        num_cs_cum = np.cumsum(num_cs)

        # Concat into C channel and do augmentations
        X = torch.concat(tensors, dim=1)  # TCHW

        # Normalize
        X = X.to(dtype=torch.get_default_dtype()).div(255)  # 0~1
        
        # TODO: resize but keep ratio?
        # Resize
        X = self.trans_resize(X)

        # Optional Augment (crop&pad, color jitter)
        if self.do_augment == True:
            X = self.trans_crop(X)
            pass

        # Split from C channel
        Y = torch.tensor_split(X, num_cs_cum.tolist()[:-1], dim=1)
        return Y

    @staticmethod
    def _debug_show_img(x):
        # x shape: (T,C,H,W)
        matplotlib.use('TkAgg')
        img = x[0].numpy().transpose((1, 2, 0))
        plt.imshow(img)
        plt.show()

    def _features_from_indices(self, clip_indices, boxes, rgb_path, label):
        """
        Args:
            boxes: shape (T, P, 4)
        """
        boxes = np.array(boxes, dtype=object)

        flow_path = ChaPath(rgb_path).change_base(self.flow_base)
        uv_path = ChaPath(rgb_path).change_base(self.uv_base)

        boxes_clip = boxes[clip_indices, :]  # (T, P, 4 or None)  time indices

        box = self.compose.combine_temporal_box_xyxy(boxes_clip, self.parts)  # (4)
        x1, y1, x2, y2 = box
        x1 = max(0, x1)
        y1 = max(0, y1)
        
        # Load video, temporal clip
        flow_clip = VideoIO.read_video_TCHW(flow_path, 2, clip_indices)
        uv_clip = VideoIO.read_video_TCHW(uv_path, 2, clip_indices)
        rgb_clip = VideoIO.read_video_TCHW(rgb_path, 0, clip_indices, format='rgb24')

        # Crop part area from image
        flow_crop, uv_crop, rgb_crop = [x[:, :, y1:y2, x1:x2] for x in [flow_clip, uv_clip, rgb_clip]]

        # Augment (resize, normalizations, crop(optional))
        rgb_crop, flow_crop, uv_crop = self._preprocess([rgb_crop, flow_crop, uv_crop])

        out_dict = {
            'label': label - 1,
            'rgb': rgb_crop,  # TCHW
            'uv': uv_crop,
            'flow': flow_crop

        }

        return out_dict

    def _random_sampling(self, seq_len, clip_len):
        possible_start_idx = seq_len - clip_len
        possible_start_idx = max(0, possible_start_idx)
        start_idx = random.randint(0, possible_start_idx)  # (randint: start/end both included)
        clip_indices = range(start_idx, start_idx + clip_len)
        clip_indices = [i % seq_len for i in clip_indices]  # If clip is larger than video length, then pick from start
        return clip_indices
    
    def _uniform_sampling(self, seq_len, clip_len):
        clips = []
        if(seq_len <= clip_len):
            clips.append(self._random_sampling(seq_len, clip_len))
        else:
            t = 0
            for t in range(0, seq_len - clip_len, clip_len):
                clip_indices = range(t, t + clip_len)
                clips.append(clip_indices)
        return clips
    
    def __getitem__(self, index):

        rgb_path, depth_path, label = self.label_list[index]
        rgb_path = Path(self.root, self.sample_base, rgb_path)
        depth_path = Path(self.root, self.sample_base, depth_path)
        
        box_path = ChaPath(rgb_path).change_base(self.box_base).with_suffix('.pkl')
        # Sometimes pyav frame counts do not work, therefore loads boxes for frame count
        with box_path.open('rb') as f:
            boxes = pickle.load(f)

        # Random / Uniform sampling
        # Random sampling
        seq_len = len(boxes) - 1  # Should use seq_len, not -1, but the video reader returns OUT_OF_BOUND
        if self.sampling == 'random':
            # Random sampling, take 1
            clip_indices = self._random_sampling(seq_len, self.clip_len)
            collected_feature = self._features_from_indices(clip_indices, boxes, rgb_path, label)
            return collected_feature

        elif self.sampling == "uniform":
            # Uniform sampling, take N, N: generated with sliding window of clip_len and stride 4
            clip_indices_list = self._uniform_sampling(seq_len, self.clip_len)
            collected_feature_list = []
            for clip_indices in clip_indices_list:
                collected_feature = self._features_from_indices(clip_indices, boxes, rgb_path, label)
                collected_feature_list.append(collected_feature)
            return collected_feature_list


from model.my_slowfast import init_my_slowfast
class ModelManager():
    def __init__(self, cfg) -> None:
        self.model = init_my_slowfast(cfg, (5, 2), (64, 8))
        self.model = self._load_pretrain(self.model)
        self.ckpt_dir = Path(cfg.CHALEARN.ROOT, cfg.MODEL.LOGS, cfg.MODEL.CKPT_DIR, cfg.MODEL.NAME)
        self.debug = cfg.DEBUG

    def _load_pretrain(self, model):
        
        pretrained = torch.load(Path('pretrained', 'SLOWFAST_8x8_R50.pyth'))
        state_dict = pretrained["model_state"]

        state_dict = self._delete_mismatch(state_dict)

        model.load_state_dict(state_dict, strict=False)
        model.cuda()
        return model
    
    def prepare_data(self, batch:dict):
        # batch shape: NTCHW

        # NTCHW -> NCTHW
        x = [torch.permute(x, [0, 2, 1, 3, 4]).cuda() for x in (batch['rgb'], batch['uv'], batch['flow'])]
        x = [torch.cat([x[0], x[1]], dim=1), x[2]]
        y_true = batch['label'].cuda()

        return x, y_true
    
    def _delete_mismatch(self, state_dict):
        layers = [
            'blocks.0.multipathway_blocks.0.conv.weight',
            'blocks.0.multipathway_blocks.1.conv.weight',
            'blocks.6.proj.weight',
            'blocks.6.proj.bias',

            'blocks.1.multipathway_blocks.0.res_blocks.0.branch1_conv.weight',
            'blocks.1.multipathway_blocks.0.res_blocks.0.branch2.conv_a.weight',
            'blocks.2.multipathway_blocks.0.res_blocks.0.branch1_conv.weight',
            'blocks.2.multipathway_blocks.0.res_blocks.0.branch2.conv_a.weight',
            'blocks.3.multipathway_blocks.0.res_blocks.0.branch1_conv.weight',
            'blocks.3.multipathway_blocks.0.res_blocks.0.branch2.conv_a.weight',
            'blocks.4.multipathway_blocks.0.res_blocks.0.branch1_conv.weight',
            'blocks.4.multipathway_blocks.0.res_blocks.0.branch2.conv_a.weight',
        ]
        for key in layers:
            del state_dict[key]
        return state_dict

    def save_ckpt(self, epoch=0, acc=0.0):
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_name = 'acc%.3f_e%d.ckpt' % (acc, epoch)
        # ckpt_name = f'acc{round(acc, 2)}_e{epoch}.ckpt'
        ckpt_path = Path(self.ckpt_dir, ckpt_name)
        
        if not self.debug:
            torch.save(self.model.state_dict(), ckpt_path)
            print(f"Checkpoint saved in {str(ckpt_path)}")
        else:
            print(f'Ignore checkpoint saving under debug mode. {str(ckpt_path)}')
        

    def load_ckpt(self):
        ckpt_list = sorted(glob.glob(str(self.ckpt_dir / '*.ckpt')))
        if len(ckpt_list) == 0:
            print('warning: no checkpoint found, try with HTAH ckeckpoint')
            # Use HTAH checkpoint:
            ckpt_list = sorted(glob.glob(str(Path(self.ckpt_dir.parent, 'slowfast-HTAH', '*.ckpt'))))
            if len(ckpt_list) == 0:
                print('warning: no HTAH checkpoint found')
                return
        ckpt = ckpt_list[-1]
        print(f'loading checkpoint from {str(ckpt)}')
        
        state_dict = torch.load(ckpt)
        # self.mm.delete_mismatch(state_dict)
        self.model.load_state_dict(state_dict, strict=True)

        pass

from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import default_collate

class Trainer():

    def __init__(self, cfg):
        self.debug = cfg.DEBUG

        if self.debug == True:
            self.num_workers = 0
            self.save_debug_img = True
            
        elif self.debug == False:
            self.num_workers = min(cfg.NUM_CPU, 10)
            self.save_debug_img = False
        
        self.cfg = cfg
        self.batch_size = cfg.CHALEARN.BATCH_SIZE

        self.parts = PartCompose.lHandArmTorso
            
        self.train_dataset = ChalearnGestureDataset(cfg, 'train', self.parts, 'random', do_augment=False)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=cfg.CHALEARN.BATCH_SIZE, shuffle=True, drop_last=False, num_workers=self.num_workers)

        # self.valid_dataset = ChalearnVideoDataset(cfg, 'valid')
        # self.valid_loader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=cfg.CHALEARN.BATCH_SIZE, shuffle=False, drop_last=True, num_workers=self.num_workers)

        self.test_dataset = ChalearnGestureDataset(cfg, 'test', self.parts, 'uniform', do_augment=False)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=cfg.CHALEARN.BATCH_SIZE, shuffle=False, drop_last=False, num_workers=self.num_workers, collate_fn=lambda x:x)

        self.mm = ModelManager(cfg)

        self.loss = CrossEntropyLoss()
        
        self.num_step = 0
        self.max_historical_acc = 0.

        self.mm.load_ckpt()

        self.optim = optim.SGD(self.mm.model.parameters(), lr=cfg.MODEL.LR, momentum=0.9)

    def _train_epoch(self):

        loss_list = []
        correct_list = []
        for batch in tqdm(self.train_loader):
            # batch: dict of NTCHW, except for labels
            
            x, y_true = self.mm.prepare_data(batch)  # x: list of N,T,C,H,W

            self.mm.model.train()
            y_pred = self.mm.model(x)

            loss_tensor = self.loss(y_pred, y_true)
            self.optim.zero_grad()
            loss_tensor.backward()
            self.optim.step()

            # if self.num_step % 100 == 0:
            #     print(f'Step {self.num_step}, loss: {round(loss_tensor.item(), 3)}')
            self.num_step = self.num_step + 1 
            loss_list.append(loss_tensor.item())

            # Compute train correctness
            with torch.no_grad():
                y_pred = torch.argmax(y_pred, dim=-1)
                correct = y_pred == y_true
                correct_list.append(correct)
            
            if self.debug:
                break

        loss_avg = np.array(loss_list).mean()
        print(f'loss_avg: {round(loss_avg, 3)}')

        c = torch.concat(correct_list, dim=0)
        accuracy = c.sum() / len(c)
        print(f'Train Accuracy: {round(accuracy.item(), 3)}. ({c.sum().item()} / {len(c)})')
        
    
    def train(self):
        
        max_epoch = self.cfg.MODEL.MAX_EPOCH

        for epoch in range(max_epoch):
            print(f'========== Training epoch {epoch}')
            self.num_step = 0
            self._train_epoch()
            
            if (epoch) % 1 == 0:
                y = self.run_eval()

                acc = y['acc']
                if acc > self.max_historical_acc:
                    self.max_historical_acc = acc
                    self.mm.save_ckpt(epoch, acc)
                else:
                    print(f"Not saved. Current best acc: %.3f" % (self.max_historical_acc))
                    
        
        self.mm.save_ckpt(epoch, acc)

    # Run dataloader with eval model
    def run_eval(self, dataset_loader: torch.utils.data.DataLoader=None,):

        prepare_data = self.mm.prepare_data
        model = self.mm.model
        if dataset_loader is None:
            dataset_loader = self.test_loader

        pred_score_list = []  # List of (N, class_score)
        true_list = []  # List of (N,)

        batch_collect = []  # Collect a batch
        samples_per_video = []  # [7, 5, 10, ...]

        def test_batch(collect):
            
            x, y_true = prepare_data(collect)

            with torch.no_grad():
                model.eval()
                y_pred = model(x)  # N,class_score
            
            y_pred = y_pred.cpu().numpy()
            y_true = y_true.cpu().numpy()

            pred_score_list.append(y_pred)
            true_list.append(y_true)
        
        for step, batch in enumerate(tqdm(dataset_loader)):  # LNTCHW, N=1, L for list generated from dataset
            # batch: [[TCHW]]
            # How many samples are uniformly generated from the same video. These are aggregated later through majority voting
            [samples_per_video.append(len(b)) for b in batch]  # b: N,TCHW, N is length of a video - clip_len
            [batch_collect.extend(b) for b in batch]  # batch_collect: N,TCHW
            if len(batch_collect) < self.batch_size:
                continue
            
            while len(batch_collect) > self.batch_size:
                # Batch size reached. Run a batch from batch_collect
                batch_full = default_collate(batch_collect[:self.batch_size])
                batch_collect = batch_collect[self.batch_size:]
                
                test_batch(batch_full)
            
            if self.debug == True and step > 5:
                break
        
        # Last batch
        if len(batch_collect) > 0:
            batch_collect = default_collate(batch_collect)
            test_batch(batch_collect)

        pred_score_arr = np.concatenate(pred_score_list, axis=0)  # (N, class_score)
        pred_score_arr = np.exp(pred_score_arr) / np.sum(np.exp(pred_score_arr), axis=1, keepdims=True)  # (N, class_score)
        # pred_arr = np.argmax(pred_score_arr, axis=1)  # (N,)
        true_arr = np.concatenate(true_list, axis=0)  # (N,)

        # Acc 

        correct_list = []
        read_index = 0
        for num_samples in samples_per_video:
            v_begin = read_index
            v_end = read_index + num_samples
            read_index = read_index + num_samples
            preds = pred_score_arr[v_begin: v_end]  # (N, class_score)
            preds = np.mean(preds, axis=0)  # Mean over N 
            trues = true_arr[v_begin: v_end]
            assert np.all(trues == trues[0])

            # most_pred = np.bincount(preds).argmax()
            pred_1 = np.argmax(preds, axis=0)
            true = trues[0]
            is_correct = (pred_1 == true)

            correct_list.append(is_correct)
            
        c = np.array(correct_list)
        accuracy = c.sum() / len(c)
        print(f'Test Accuracy: {round(accuracy, 3)}. ({c.sum()} / {len(c)})')
        return {
            'ps':pred_score_arr,
            't': true_arr,
            'acc': accuracy,
            'sv': samples_per_video,
        }


if __name__ == '__main__':

    from config.defaults import get_override_cfg
    _cfg = get_override_cfg()
    # ConvertVideoToFlow(_cfg).convert()
    # ConvertVideoToIUVPkl(_cfg).convert()
    # ConvertIuvPklToUvVideo(_cfg).convert()
    # ConvertIuvPklToPartBox(_cfg).convert()
    # ConvertIuvPklToPartBox(_cfg).plot()
    # dataset = ChalearnGestureDataset(_cfg, 'train', PartCompose.TorsoArmHand, 'random')
    # dataset[5]
    Trainer(_cfg).train()
