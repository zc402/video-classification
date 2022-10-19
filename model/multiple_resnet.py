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
from torch.nn import CrossEntropyLoss, Module, Linear, Conv2d, Conv3d, Identity
from torch import optim
from config.defaults import get_override_cfg

from dataset.chalearn_dataset import ChalearnVideoDataset

cfg = get_override_cfg()

class ResnetWrapper(Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)  # resnet50
        self.resnet.conv1 = Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # a channel produces a vector of softmax scores of length C, where C is the number of classes
        conv_out_channels = self.resnet.fc.in_features
        self.resnet.fc = Linear(conv_out_channels, out_channels, bias=True)  # Overwrite the original fc classifier as we want fusions later
        
    def forward(self, x):
        x = self.resnet(x)  # N,C
        return x  # output shapes depend on the resnet:  resnet 18: 512. resent50: 2048 

class MultipleResnet(Module):

    def __init__(self, in_channels_list):
        """
        in_channels_list: list of num_channel for each resnet. e.g. [3,3,3,2,2,2]
        """
        super().__init__()
        self.num_class = cfg.CHALEARN.SAMPLE_CLASS
        self.N = cfg.CHALEARN.BATCH_SIZE
        self.T = cfg.CHALEARN.CLIP_LEN

        # number of input channel = 
        self.resnet_list = torch.nn.ModuleList([ResnetWrapper(n*self.T, self.num_class) for n in in_channels_list])

        # sparsely connected network with one weight per gesture × channel
        # self.fc = torch.nn.ModuleList([Linear(self.resnet_out_channels, self.num_class) for i in range(num_resnet)])
        # self.resnet_out_channels = self.resnet_list[0].out_channels
        # self.fc = Linear(self.resnet_out_channels, self.num_class)
    
    def forward(self, x_s):
        # x_s: list of x
        # x.size(): NTCHW
        y_pred_list = []
        for i, x in enumerate(x_s):
            N,T,C,H,W = x.size()
            x = torch.reshape(x, (N, T * C, H, W))  # Stack time dim into color channels
            y_pred = self.resnet_list[i](x)  # N,channels
            # y_pred = torch.reshape(y_pred, (N, T, self.resnet_out_channels))  # N, T, C
            # y_pred = torch.mean(y_pred, dim=1)  # Average time dim for each human part channel

            y_pred_list.append(y_pred)

        y = torch.stack(y_pred_list, dim=0)  # Part, N, Class
        y = torch.mean(y, dim=0)  # N,Class
        # y = self.fc(y)  # N, class
        return y

    def _NTCHW_to_NCHW(self, x):
        N,T,C,H,W = x.size()
        x = torch.reshape(x, (N*T, C,H,W))
        return x