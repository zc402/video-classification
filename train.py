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
from config.crop_cfg import crop_folder_list

class Trainer():

    def __init__(self):
        debug = False
        if debug:
            self.num_workers = 0
            self.save_debug_img = True
        else:
            self.num_workers = 12
            self.save_debug_img = False
        self.train_dataset = ChalearnVideoDataset('train')
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=cfg.CHALEARN.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=self.num_workers)

        self.test_dataset = ChalearnVideoDataset('test')
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=cfg.CHALEARN.BATCH_SIZE, shuffle=False, drop_last=True, num_workers=self.num_workers)
    
        self.model = None
        self.loss = CrossEntropyLoss()
        self.optim = None
        self.num_step = 0
        self.ckpt_dir = Path(cfg.MODEL.CKPT_DIR)
        self.num_class = cfg.CHALEARN.SAMPLE_CLASS
        # self.save_debug_img = False  # Save batch data for debug

    def _lazy_init_model(self, in_channels, num_resnet):
        self.model = MultipleResnet(in_channels, num_resnet).cuda()
        self.model.train()

        self.optim = optim.Adam(self.model.parameters(), lr=1e-3)

        if self.ckpt_dir.exists():
            self.load_ckpt()
        else:
            print('warning: checkpoint not found!')

    def save_ckpt(self, epoch=0, acc=0.0):
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_name = f'acc-{round(acc, 2)}_e-{epoch}.ckpt'
        ckpt_path = Path(self.ckpt_dir, ckpt_name)
        torch.save(self.model.state_dict(), ckpt_path)
        print(f"Checkpoint saved in {str(ckpt_path)}")

    def load_ckpt(self):
        # state_dict = torch.load(self.ckpt_dir)
        # self.model.load_state_dict(state_dict)
        pass


    def prepare_data(self, batch):
        batch = {k: x.cuda() for k, x in batch.items()}
        image_features = [v for k, v in batch.items() if k in crop_folder_list]
        y_true = batch['label']

        if self.save_debug_img:
            self.debug_show(batch['CropHTAH'])  # NTCHW     
        return image_features, y_true

    def epoch(self):

        for batch in tqdm(self.train_loader):
            # batch: dict of NTCHW, except for labels
            
            x, y_true = self.prepare_data(batch)  # x: list of (N,T,)C,H,W

            if self.model is None:
                N,T,C,H,W = batch['CropHTAH'].shape
                num_resnet = len(x)
                print(f'Construct {num_resnet} resnet channels')
                self._lazy_init_model(in_channels=C, num_resnet=num_resnet)
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
        
        for epoch in range(40):
            print(f'Epoch {epoch}')
            self.num_step = 0
            self.epoch()
            acc = self.test()
            self.save_ckpt(epoch, acc)
    
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
        return accuracy
    
    def debug_show(self, input):
        debug_folder = Path('logs', 'debug')
        debug_folder.mkdir(parents=True, exist_ok=True)
        # NTCHW
        for frame in range(0, cfg.CHALEARN.CLIP_LEN):
            x = input[0][frame].cpu().numpy()
            x = np.transpose(x, (1,2,0))  # HWC, C: BGRUV
            U = x[:, :, 3]
            plt.imshow(U)
            plt.savefig(Path(debug_folder, f'S{str(self.num_step).zfill(2)}_T{str(frame).zfill(2)}'))
            plt.close()
            print('image saved')

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
