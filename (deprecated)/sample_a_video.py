import pytorchvideo.data
import torch
from fractions import Fraction
import numpy as np
import cv2
from matplotlib import pyplot as plt
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip
)

plt.ion()

CLIP_DURATION = 8 # Fraction(15, 10)

# CHA_LEARN_ROOT_PATH = '/home/zc/Datasets/[手势识别]ChaLearn/ChaLearn Isolated Gesture Recognition/train/'
CHA_LEARN_ROOT_PATH = '/home/zc/Datasets/ChaLearnIsoConst/train/'

video_list = [
    (CHA_LEARN_ROOT_PATH + "train/001/M_00001.avi", {'label': 26}),
    (CHA_LEARN_ROOT_PATH + "train/001/M_00002.avi", {'label': 19}),
    (CHA_LEARN_ROOT_PATH + "train/001/M_00003.avi", {'label': 20}),
]

transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(8),
                            Lambda(lambda x: x / 255.0),
                            Normalize((0.45, 0.45, 0.45),
                                      (0.225, 0.225, 0.225)),
                            # RandomShortSideScale(min_size=256, max_size=320),
                            # RandomCrop(244),
                            # RandomHorizontalFlip(p=0.5),
                        ]
                    ),
                ),
            ]
        )

train_dataset = pytorchvideo.data.LabeledVideoDataset(
    video_list, 
    clip_sampler=pytorchvideo.data.make_clip_sampler("random", CLIP_DURATION),
    decode_audio=False,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=0,
    )

for data in train_loader:
    print(data['label'])
    print(data['video'].shape)
    video = data['video'][0]
    # video = video.numpy()  # C,T,H,W
    # for i in range(video.shape[1]):
    #     frame = video[:, i]
    #     frame = frame.astype('int')
    #     frame = np.transpose(frame, (1,2,0))
    #     plt.imshow(frame)
    #     plt.show()
    #     plt.pause(0.1)

    pass

pass