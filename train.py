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
cfg = get_override_cfg()

from dataset.chalearn_dataset import ChalearnVideoDataset
from model.multiple_resnet import MultipleResnet

class Trainer():

    def __init__(self):
        self.train_dataset = ChalearnVideoDataset('train')
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=cfg.CHALEARN.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)

        self.test_dataset = ChalearnVideoDataset('test')
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=cfg.CHALEARN.BATCH_SIZE, shuffle=False, drop_last=True, num_workers=4)
    
        self.model = MultipleResnet().cuda()
        self.loss = CrossEntropyLoss()
        self.optim = optim.Adam(self.model.parameters(), lr=1e-3)
        self.num_step = 0
        self.ckpt = Path(cfg.MODEL.CKPT)
        self.num_class = cfg.CHALEARN.SAMPLE_CLASS

    def save_ckpt(self):
        self.ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.ckpt)
        print("Save checkpoint")

    def load_ckpt(self):
        state_dict = torch.load(self.ckpt)
        self.model.load_state_dict(state_dict)


    def prepare_data(self, batch):
        y_true = batch['label'].cuda()

        large = ['CropBody', 'CropTorsoLArm', 'CropTorsoRArm']
        medium = ['CropLHandArm', 'CropRHandArm']
        small = ['CropLHand', 'CropRHand', ]

        white_list = large + medium + small
        batch = {k:v for k,v in batch.items() if k in white_list}
        batch = {k: x.cuda() for k, x in batch.items()}

        # self.debug_show(x)        
        return batch.values(), y_true

    def epoch(self):

        for batch in tqdm(self.train_loader):

            """# batch['CropLHand'].size() : [N, T, C, H, W]
            crop_keys = [key for key in batch.keys() if 'Crop' in key]
            crop_imgs = [batch[key] for key in crop_keys]  
            crop_imgs = torch.stack(crop_imgs)  # shape: KNTCHW, K: crop_key
            # --> (NT)(KC)HW as NCHW expected by resnet
            K,N,T,C,H,W = crop_imgs.size()
            crop_imgs = torch.permute(crop_imgs, (1,2,0,3,4,5,))  # NTKCHW
            crop_imgs = torch.reshape(crop_imgs, (N*T, K*C, H, W,))  """
            
            x, y_true = self.prepare_data(batch)

            y_pred = self.model(x)


            loss_tensor = self.loss(y_pred, y_true)
            self.optim.zero_grad()
            loss_tensor.backward()
            self.optim.step()

            if self.num_step % 100 == 0:
                print(f'Step {self.num_step}, loss: {round(loss_tensor.item(), 3)}')
            self.num_step = self.num_step + 1 
        
    
    def train(self):
        self.model.train()
        if self.ckpt.exists():
            self.load_ckpt()
        else:
            print('warning: checkpoint not found!')
        
        for epoch in range(20):
            print(f'Epoch {epoch}')
            self.num_step = 0
            self.epoch()
            # self.save_ckpt()
            self.test()
    
    def test(self):
        self.model.eval()  # Temporally switch to eval mode
        print("Testing ...")
        correct_list = []
        for batch in tqdm(self.test_loader):

            x, y_true = self.prepare_data(batch)
            with torch.no_grad():
                y_pred = self.model(x)  # N,class_score
            y_pred = torch.argmax(y_pred, dim=-1)
            correct = y_pred == y_true
            correct = correct.cpu().numpy()
            correct_list.append(correct)
        c = np.concatenate(correct_list, axis=0)
        accuracy = c.sum() / len(c)
        print(f'Accuracy: {round(accuracy, 2)}. ({c.sum()} / {len(c)})')
        self.model.train()  # Recover to training mode
    
    def debug_show(self, x):
        # NCHW
        x = x[5].cpu().numpy()
        x = np.transpose(x, (1,2,0))  # HWC
        plt.imshow(x)
        plt.savefig(Path('debug', str(self.num_step)))
        plt.close()

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
