from pathlib import Path
from random import random
import torch
import torch.utils.data 
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import warnings
from torch.nn import CrossEntropyLoss, Module, Linear, Conv2d
from torch import optim
from config.defaults import get_override_cfg

from dataset.chalearn_dataset import ChalearnVideoDataset

cfg = get_override_cfg()

class ResnetWrapper(Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.resnet.conv1 = Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = Linear(512, out_channels)
    
    def forward(self, x):
        x = self.resnet(x)  # N,C
        return x

class MultipleResnet(Module):

    def __init__(self):
        super().__init__()
        self.num_class = cfg.CHALEARN.SAMPLE_CLASS
        self.N = cfg.CHALEARN.BATCH_SIZE
        self.T = cfg.CHALEARN.CLIP_LEN
        self.resnet_out_channels = 512
        num_resnet = 7

        self.resnet_list = torch.nn.ModuleList([ResnetWrapper(3, self.resnet_out_channels) for i in range(num_resnet)])
        [model.cuda() for model in self.resnet_list]

        self.fc = torch.nn.ModuleList([Linear(self.resnet_out_channels, self.num_class) for i in range(num_resnet)])
    
    def forward(self, x_s):
        # x_s: list of x
        # x.size(): NTCHW
        y_pred_list = []
        for i, x in enumerate(x_s):
            N,T,C,H,W = x.size()
            x = torch.reshape(x, (N*T, C,H,W))
            y_pred = self.resnet_list[i](x)  # N,channels
            y_pred = torch.reshape(y_pred, (N, T, self.resnet_out_channels))  # NC
            y_pred = self.fc[i](y_pred)  #N,Class
            y_pred_list.append(y_pred)

        # how much weight to assign to a channel, given a gesture
        y = torch.stack(y_pred_list, dim=0)
        y = torch.sum(y, dim=0)  # N,T,Class
        # Mean over time dim
        y = torch.mean(y, dim=1)  # N,Class
        return y

    def _NTCHW_to_NCHW(self, x):
        N,T,C,H,W = x.size()
        x = torch.reshape(x, (N*T, C,H,W))
        return x