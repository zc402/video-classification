import glob
import pickle
import sys
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
    def __init__(self) -> None:
        from config.defaults import get_override_cfg
        cfg = get_override_cfg()
        self.video_sample_root = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.SAMPLE)
        self.flow_base = '2_Flow_New'
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
    def __init__(self) -> None:
        self.iuv_base = '4_IUV_New'
        from config.defaults import get_override_cfg
        cfg = get_override_cfg()
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
            iuv_pkl_path = Path(iuv_pkl_path.parent, iuv_pkl_path.stem + '.pkl')
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
    def __init__(self) -> None:
        self.iuv_base = '4_IUV_New'
        self.uv_vid_base = '5_UV_Video'

        self.y_pad = 120  # y-pad-left
        self.x_pad = 160  # x-pad-left
        
        self.img_h = 240
        self.img_w = 320

        self.img_pad_h = 240*2
        self.img_pad_w = 320*2
        

        from config.defaults import get_override_cfg
        cfg = get_override_cfg()
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
        uv_vid_path = Path(uv_vid_path.parent, uv_vid_path.stem + '.avi')
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
    def __init__(self) -> None:
        self.iuv_base = '4_IUV_New'

        self.y_pad = 120  # y-pad-left
        self.x_pad = 160  # x-pad-left
        
        self.img_h = 240
        self.img_w = 320
        
        self.num_parts = 25  # 0~24, 0 is background
        from config.defaults import get_override_cfg
        cfg = get_override_cfg()

        self.box_base = '6_Box'
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
        video_path = video_path.parent / (video_path.stem + '.avi')

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


class PartBoxComposition:

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
        x1 = box_arr[:, 0]
        y1 = box_arr[:, 1]
        x2 = box_arr[:, 2]
        y2 = box_arr[:, 3]

        x1_min = min(x1)
        x2_min = min(x2)
        y1_max = max(y1)
        y2_max = max(y2)
        large_box = (x1_min, y1_max, x2_min, y2_max)
        return large_box

    def combine_spatial_box_xyxy(self, part_boxes, part_list):
        """
        Args:
            part_boxes: list of boxes, shape [P][4]
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

    def __init__(self) -> None:
        pass

# matplotlib.use('TkAgg')
# plt.imshow(video[10])
# plt.show()
if __name__ == '__main__':
    # ConvertVideoToFlow().convert()
    # ConvertVideoToIUVPkl().convert()
    # ConvertIuvPklToUvVideo().convert()
    # ConvertIuvPklToPartBox().convert()
    ConvertIuvPklToPartBox().plot()