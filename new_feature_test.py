import glob
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

    @staticmethod
    def read_video_TCHW(filename: Path, channels: int):
        videos = []
        for c in range(channels):
            cPath = Path(filename.parent, f'{c}_{filename.name}')
            video = VideoIO.read_video_gray(cPath)
            videos.append(video)  # THW
        videos = torch.stack(videos, dim=1)  # TCHW

        return videos
    
    @staticmethod
    def read_video(filename: Path, format='gray'):
        frame_list = []
        with av.open(filename) as container:
            for frame in container.decode(video=0):
                frame_list.append(frame.to_ndarray(format=format))

        video_array = np.stack(frame_list)
        return video_array
    
    def read_video_rgb(filename: Path):
        return VideoIO.read_video(filename, format='rgb24')
    
    def read_video_gray(filename: Path):
        return VideoIO.read_video(filename, format='gray')


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

        video = VideoIO.read_video_rgb(filename)
        T, H, W, C = video.shape
        
        flow_list = []
        img1_batch = []
        img2_batch = []
        for t in range(T-1):
            img1_batch.append(video[t])
            img2_batch.append(video[t+1])

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

class ConvertVideoToIUV:
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
            iuv_file = ChaPath(avi).change_base(self.iuv_base)
            iuv_file = Path(iuv_file.parent, iuv_file.stem + '.pkl')
            # if iuv_file.exists():
            #     continue
            iuv_file.parent.mkdir(parents=True, exist_ok=True)

            self.apply_net.entrance(
                self.yaml_path, 
                self.model_pkl,
                avi,
                iuv_file.as_posix())

# matplotlib.use('TkAgg')
# plt.imshow(video[10])
# plt.show()
if __name__ == '__main__':
    ConvertVideoToIUV().convert()