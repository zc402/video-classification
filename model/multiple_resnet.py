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
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).cuda()
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

        self.r_lg = ResnetWrapper(3*3, self.resnet_out_channels)  # Large crops
        self.r_md = ResnetWrapper(3*4, self.resnet_out_channels)  
        self.r_sm = ResnetWrapper(3*2, self.resnet_out_channels)  

        self.fc = Linear(self.resnet_out_channels*2, self.num_class)
    
    def forward(self, x):
        x1, x2, x3 = x
        x1 = self.r_lg(x1)
        x1 = torch.reshape(x1, (self.N, self.T, self.resnet_out_channels))  # N,T,Out

        x2 = self.r_md(x2)
        x2 = torch.reshape(x2, (self.N, self.T, self.resnet_out_channels))

        x3 = self.r_sm(x3)
        x3 = torch.reshape(x3, (self.N, self.T, self.resnet_out_channels))

        x = torch.concat([x1, x2], dim=-1)
        x = self.fc(x)

        x = torch.mean(x, dim=1)  # N,C
        return x