import os
from pathlib import Path
from yacs.config import CfgNode as CN
from fractions import Fraction

import torch
import pytorch_lightning
import pytorchvideo.data
import torch.utils.data
import pytorchvideo.models.resnet
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip
)
from matplotlib import pyplot as plt
from config.defaults import get_cfg

plt.ion()

# 错误的标注： torch.Size([1, 3, 2, 240, 320]) val['M_05257.avi']
# 最短的val视频：16帧


side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32
sampling_rate = 2
frames_per_second = 30
alpha = 4

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(
                size=side_size
            ),
            # CenterCropVideo(crop_size),
            PackPathway()
        ]
    ),
)

class ChalearnDataModule(pytorch_lightning.LightningDataModule):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.chalearn_root = cfg.CHALEARN.ROOT

        self.train_folder = Path(self.chalearn_root, 'train')
        self.train_txt = Path(self.chalearn_root, 'IsoGD_labels', 'train.txt')

        self.val_folder = Path(self.chalearn_root, 'valid')
        self.val_txt = Path(self.chalearn_root, 'IsoGD_labels', 'valid.txt')

        self.test_folder = Path(self.chalearn_root, 'test')
        self.test_txt = Path(self.chalearn_root, 'IsoGD_labels', 'test.txt')

        self.batch_size = cfg.CHALEARN.BATCH_SIZE

    # Dataset configuration

    _CLIP_DURATION = 8  # Fraction(15, 10)  # Duration of sampled clip for each video, in seconds
    _NUM_WORKERS = 4  # Number of parallel processes fetching data

    def train_dataloader(self):
        """
        Create the train partition from the list of video labels
        """

        # train_transform = Compose(
        #     [
        #         ApplyTransformToKey(
        #             key="video",
        #             transform=Compose(
        #                 [
        #                     UniformTemporalSubsample(8),
        #                     Lambda(lambda x: x / 255.0),
        #                     Normalize((0.45, 0.45, 0.45),
        #                               (0.225, 0.225, 0.225)),
        #                     # RandomShortSideScale(min_size=256, max_size=320),
        #                     # RandomCrop(244),
        #                     # RandomHorizontalFlip(p=0.5),
        #                 ]
        #             ),
        #         ),
        #     ]
        # )

        m_labels = self._convert_labeled_video_paths(
            self.train_txt, self.train_folder)  # (visual modality video path, label)

        train_dataset = pytorchvideo.data.LabeledVideoDataset(
            labeled_video_paths=m_labels,
            clip_sampler=pytorchvideo.data.make_clip_sampler(
                "random", self._CLIP_DURATION),
            decode_audio=False,
            transform=transform
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self._NUM_WORKERS,
        )

    # def val_dataloader(self):
    #     """
    #     Create the Kinetics validation partition from the list of video labels
    #     in {self._DATA_PATH}/val
    #     """

    #     val_transform = Compose(
    #         [
    #             ApplyTransformToKey(
    #                 key="video",
    #                 transform=Compose(
    #                     [
    #                         UniformTemporalSubsample(8),
    #                         Lambda(lambda x: x / 255.0),
    #                         Normalize((0.45, 0.45, 0.45),
    #                                   (0.225, 0.225, 0.225)),
    #                         # RandomShortSideScale(min_size=256, max_size=320),
    #                         # RandomCrop(244),
    #                         # RandomHorizontalFlip(p=0.5),
    #                     ]
    #                 ),
    #             ),
    #         ]
    #     )

    #     m_labels = self._convert_labeled_video_paths(self.val_txt, self.val_folder)

    #     val_dataset = pytorchvideo.data.LabeledVideoDataset(
    #         labeled_video_paths=m_labels,
    #         clip_sampler=pytorchvideo.data.make_clip_sampler(
    #             "uniform", self._CLIP_DURATION,),
    #         decode_audio=False,
    #         transform=val_transform
    #     )
    #     return torch.utils.data.DataLoader(
    #         val_dataset,
    #         batch_size=self.batch_size,
    #         num_workers=self._NUM_WORKERS,
    #     )

    def test_dataloader(self):

        # test_transform = Compose(
        #     [
        #         ApplyTransformToKey(
        #             key="video",
        #             transform=Compose(
        #                 [
        #                     UniformTemporalSubsample(8),
        #                     Lambda(lambda x: x / 255.0),
        #                     Normalize((0.45, 0.45, 0.45),
        #                               (0.225, 0.225, 0.225)),
        #                     # RandomShortSideScale(min_size=256, max_size=320),
        #                     # RandomCrop(244),
        #                     # RandomHorizontalFlip(p=0.5),
        #                 ]
        #             ),
        #         ),
        #     ]
        # )

        m_labels = self._convert_labeled_video_paths(self.test_txt, self.test_folder)

        test_dataset = pytorchvideo.data.LabeledVideoDataset(
            labeled_video_paths=m_labels,
            clip_sampler=pytorchvideo.data.make_clip_sampler(
                "uniform", self._CLIP_DURATION,),
            decode_audio=False,
            transform=transform
        )
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self._NUM_WORKERS,
        )

    def _convert_labeled_video_paths(self, chalearn_txt: Path, base_path: Path):
        with chalearn_txt.open('r') as f:
            labels = f.readlines()  # line format: 'M K L'

        labels = [line.split(' ') for line in labels]  # [M, K, L]
        # new_labels = [(str(Path(base_path, m)), {'label':int(l)}) for (m, k, l) in labels]  # (M, L)
        new_labels = [(str(Path(base_path, m)), {'label':int(l)}) for (m, k, l) in labels if int(l) <= 10]
        return new_labels


# def make_kinetics_resnet():
#     return pytorchvideo.models.resnet.create_resnet(
#         input_channel=3,  # RGB input
#         model_depth=50,  # For the tutorial let's just use a 50 layer network
#         # model_num_class=cfg.CHALEARN.NUM_CLASS + 1,
#         model_num_class=250,
#         norm=nn.BatchNorm3d,
#         activation=nn.ReLU,
#     )

def slow_fast():
    model_name = "slowfast_r50"
    model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=True)
    model = model.to('cuda')
    return model

class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = slow_fast()

        self.pred_mask = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # The model expects a video tensor of shape (B, C, T, H, W), which is the
        # format provided by the dataset
        y_hat = self.model(batch["video"])

        # Compute cross entropy loss, loss.backwards will be called behind the scenes
        # by PyTorchLightning after being returned from this method.
        loss = F.cross_entropy(y_hat, batch["label"])

        # Log the train loss to Tensorboard
        self.log("train_loss", loss.item())

        return loss

    # def validation_step(self, batch, batch_idx):
    #     y_hat = self.model(batch["video"])
    #     loss = F.cross_entropy(y_hat, batch["label"])
    #     self.log("val_loss", loss)
    #     return loss

    def test_step(self, batch, batch_idx):
        y_hat = self.model(batch["video"])
        
        true_preds = (y_hat.argmax(dim=1) == batch['label'])
        print(true_preds)
        true_preds = true_preds.cpu().numpy()
        if self.pred_mask is None:
            self.pred_mask = true_preds
        else:
            self.pred_mask = np.concatenate((self.pred_mask, true_preds), axis=0)
        # loss = F.cross_entropy(y_hat, batch["label"])
        return true_preds
    
    def test_epoch_end(self, outputs) -> None:
        num_true = np.sum(self.pred_mask)
        num_all = self.pred_mask.shape[0]
        accuracy = num_true / num_all
        print(f'Accuracy: {accuracy}')
        # print(len(self.pred_mask == True) / len(self.pred_mask))
        

    def configure_optimizers(self):
        """
        Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
        usually useful for training video models.
        """
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def train():
    cfg = get_cfg()
    override = Path('..', 'cfg_override.yaml')
    if(override.is_file()):
        cfg.merge_from_file(override)
    classification_module = VideoClassificationLightningModule()
    classification_module = classification_module.load_from_checkpoint('lightning_logs/version_3/checkpoints/epoch=12-step=6032.ckpt')
    
    data_module = ChalearnDataModule(cfg)
    trainer = pytorch_lightning.Trainer(accelerator='gpu', devices=1)
    trainer.fit(classification_module, data_module)

def debug():
    cfg = get_cfg()
    override = Path('..', 'cfg_override.yaml')
    if(override.is_file()):
        cfg.merge_from_file(override)
        
    data_module = ChalearnDataModule(cfg)
    for batch in data_module.train_dataloader():
        # print(batch['label'])
        video = batch['video']
        print(video)
        if(batch['video'].shape[2] != 8):
            print("Error")
        
        # if batch['video'].size()[2] != 15:
        #     print(batch['video'].size())
        #     print(batch['video_name'])
        #     for i in range(batch['video'].size()[2]):  #video: NCTHW
        #         frame = batch['video'][0, :, i]
        #         frame = frame.cpu().numpy().transpose((1,2,0))
        #         plt.title('title')
        #         plt.imshow(frame)
        #         plt.draw()
        #         plt.pause(0.1)
        #         plt.gca().cla()

def test():
    cfg = get_cfg()
    override = Path('..', 'cfg_override.yaml')
    if(override.is_file()):
        cfg.merge_from_file(override)

    trainer = pytorch_lightning.Trainer(accelerator='gpu', devices=1)
    classification_module = VideoClassificationLightningModule()
    classification_module = classification_module.load_from_checkpoint('lightning_logs/version_3/checkpoints/epoch=12-step=6032.ckpt')
    classification_module.eval()
    data_module = ChalearnDataModule(cfg)
    trainer.test(classification_module, dataloaders=data_module.test_dataloader())

if __name__ == '__main__':
    # train()
    # debug()
    test()