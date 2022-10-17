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
from torch.nn import CrossEntropyLoss, Module, Linear, Conv2d, Conv3d
from torch import optim
from config.defaults import get_override_cfg
cfg = get_override_cfg()

from dataset.chalearn_dataset import ChalearnVideoDataset
# from model.multiple_resnet import MultipleResnet
from config.crop_cfg import crop_folder_list

class Trainer():

    def __init__(self):
        debug = False
        if debug:
            self.num_workers = 0
            self.save_debug_img = True
        else:
            self.num_workers = 10
            self.save_debug_img = False
        self.train_dataset = ChalearnVideoDataset('train')
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=cfg.CHALEARN.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=self.num_workers)

        self.test_dataset = ChalearnVideoDataset('test')
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=cfg.CHALEARN.BATCH_SIZE, shuffle=False, drop_last=True, num_workers=self.num_workers)
    
        self.model = None
        self.loss = CrossEntropyLoss()
        self.optim = None
        self.num_step = 0
        self.ckpt = Path(cfg.MODEL.CKPT)
        self.num_class = cfg.CHALEARN.SAMPLE_CLASS
        self.save_debug_img = False  # Save batch data for debug

    def _lazy_init_model(self, in_channels, num_resnet):
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        self.model.blocks[0].conv = Conv3d(2, 64, (1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.model.train()
        self.model.cuda()

        self.optim = optim.Adam(self.model.parameters(), lr=1e-3)

        if self.ckpt.exists():
            self.load_ckpt()
        else:
            print('warning: checkpoint not found!')

    def save_ckpt(self):
        self.ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.ckpt)
        print("Save checkpoint")

    def load_ckpt(self):
        state_dict = torch.load(self.ckpt)
        self.model.load_state_dict(state_dict)


    def prepare_data(self, batch):
        batch = {k: x.cuda() for k, x in batch.items()}
        image_features = [v for k, v in batch.items() if k in crop_folder_list]
        y_true = batch['label']

        if self.save_debug_img:
            self.debug_show(batch['CropHTAH'])        
        return image_features, y_true

    def epoch(self):

        for batch in tqdm(self.train_loader):
            # batch: dict of NTCHW, except for labels
            
            # x, y_true = self.prepare_data(batch)  # x: list of (N,T,)C,H,W
            x = batch['CropHTAH'][:, :, 3:].cuda()
            y_true = batch['label'].cuda()

            if self.model is None:
                N,T,C,H,W = batch['CropHTAH'].shape
                num_resnet = len(x)
                print(f'Construct {num_resnet} resnet channels')
                self._lazy_init_model(in_channels=C, num_resnet=num_resnet)
            x = torch.permute(x, [0, 2, 1, 3, 4])  # NCTHW
            y_pred = self.model(x)

            loss_tensor = self.loss(y_pred, y_true)
            self.optim.zero_grad()
            loss_tensor.backward()
            self.optim.step()

            # if self.num_step % 100 == 0:
            #     print(f'Step {self.num_step}, loss: {round(loss_tensor.item(), 3)}')
            self.num_step = self.num_step + 1 
        print(f'Step {self.num_step}, loss: {round(loss_tensor.item(), 3)}')
        
    
    def train(self):
        
        for epoch in range(50):
            print(f'Epoch {epoch}')
            self.num_step = 0
            self.epoch()
            self.save_ckpt()
            self.test()
    
    def test(self):
        self.model.eval()  # Temporally switch to eval mode
        print("Testing ...")
        correct_list = []
        for batch in tqdm(self.test_loader):

            # x, y_true = self.prepare_data(batch)
            x = batch['CropHTAH'][:, :, 3:].cuda()
            x = torch.permute(x, [0, 2, 1, 3, 4])
            y_true = batch['label'].cuda()
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
        # NTCHW
        frame = 5
        x = x[0][frame].cpu().numpy()
        x = np.transpose(x, (1,2,0))  # HWC, C: BGRUV
        U = x[:, :, 3]
        plt.imshow(U)
        plt.savefig(Path('debug', str(self.num_step).zfill(5)))
        plt.close()
        print('image saved')

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
