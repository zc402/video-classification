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
from torch.nn import CrossEntropyLoss, Module, Linear
from torch import optim
from config.defaults import get_override_cfg

from dataset.chalearn_dataset import ChalearnVideoDataset

cfg = get_override_cfg()

class ResnetWrapper(Module):

    def __init__(self):
        super().__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).cuda()
        self.resnet.fc = Linear(512, cfg.CHALEARN.SAMPLE_CLASS)
    
    def forward(self, x):
        x = self.resnet(x)
        return x


class Trainer():

    def __init__(self):
        self.dataset = ChalearnVideoDataset('train')
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=cfg.CHALEARN.BATCH_SIZE, shuffle=True)
    
        self.model = ResnetWrapper().cuda()
        self.loss = CrossEntropyLoss()
        self.optim = optim.Adam(self.model.parameters(), lr=1e-3)
        self.num_step = 0

    def epoch(self):

        for batch in tqdm(self.train_loader):
            y = batch['label'].cuda()
            """# batch['CropLHand'].size() : [N, T, C, H, W]
            crop_keys = [key for key in batch.keys() if 'Crop' in key]
            crop_imgs = [batch[key] for key in crop_keys]  
            crop_imgs = torch.stack(crop_imgs)  # shape: KNTCHW, K: crop_key
            # --> (NT)(KC)HW as NCHW expected by resnet
            K,N,T,C,H,W = crop_imgs.size()
            crop_imgs = torch.permute(crop_imgs, (1,2,0,3,4,5,))  # NTKCHW
            crop_imgs = torch.reshape(crop_imgs, (N*T, K*C, H, W,))  """

            x = batch['CropLHand']  # NTCHW
            N,T,C,H,W = x.size()
            # --> (NT)CHW
            x = torch.reshape(x, (N*T, C,H,W))

            x = x.cuda()

            y_hat = self.model(x)
            y_hat = torch.reshape(y_hat, (N,T,cfg.CHALEARN.SAMPLE_CLASS))  # N,T,class_score
            y_hat = torch.mean(y_hat, dim=1)  # N,C

            loss_tensor = self.loss(y_hat, y)
            self.optim.zero_grad()
            loss_tensor.backward()
            self.optim.step()

            if self.num_step % 20 == 0:
                print(f'Step {self.num_step}, loss {loss_tensor.item()}')
            self.num_step = self.num_step + 1 
    
    def train(self):
        
        for epoch in range(10):
            self.num_step = 0
            self.epoch()


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
