from pathlib import Path
from random import random
import torch
import torch.utils.data 
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import json
import glob
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
        if not debug:
            self.num_workers = 10
            self.save_debug_img = False
        else:  # Debug
            self.num_workers = 0
            self.save_debug_img = True
            
        self.train_dataset = ChalearnVideoDataset('train')
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=cfg.CHALEARN.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=self.num_workers)

        self.valid_dataset = ChalearnVideoDataset('valid')
        self.valid_loader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=cfg.CHALEARN.BATCH_SIZE, shuffle=False, drop_last=True, num_workers=self.num_workers)

        self.test_dataset = ChalearnVideoDataset('test')
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=False, drop_last=True, num_workers=self.num_workers)
    
        self.model = None
        self.loss = CrossEntropyLoss()
        self.optim = None
        self.num_step = 0
        self.ckpt_dir = Path(cfg.MODEL.CKPT_DIR)
        self.num_class = cfg.CHALEARN.SAMPLE_CLASS
        self.max_historical_acc = 0.
        # self.save_debug_img = False  # Save batch data for debug

    def _lazy_init_model(self, in_channel_list):
        self.model = MultipleResnet(in_channel_list).cuda()

        self.optim = optim.Adam(self.model.parameters(), lr=1e-3)

        if self.ckpt_dir.exists():
            self.load_ckpt()
        else:
            print('warning: checkpoint not found!')

    def save_ckpt(self, epoch=0, acc=0.0):
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_name = 'acc%.02f_e%d.ckpt' % (acc, epoch)
        # ckpt_name = f'acc{round(acc, 2)}_e{epoch}.ckpt'
        ckpt_path = Path(self.ckpt_dir, ckpt_name)
        torch.save(self.model.state_dict(), ckpt_path)
        print(f"Checkpoint saved in {str(ckpt_path)}")

    def load_ckpt(self):
        ckpt_list = sorted(glob.glob(str(self.ckpt_dir / '*.ckpt')))
        if len(ckpt_list) == 0:
            print('warning: no checkpoint found')
            return
        ckpt = ckpt_list[-1]
        print(f'loading checkpoint from {str(ckpt)}')
        state_dict = torch.load(ckpt)
        self.model.load_state_dict(state_dict)
        pass


    def prepare_data(self, batch):
        batch = {k: x.cuda() for k, x in batch.items()}
        # Clip C from NTCHW
        image_features_RGB = [batch[folder][:, :, 0:3] for folder in crop_folder_list]
        image_features_UV = [batch[folder][:, :, 3:5] for folder in crop_folder_list]
        image_features = image_features_RGB + image_features_UV
        y_true = batch['label']

        if self.save_debug_img:
            self.debug_show(batch['CropHTAH'])  # NTCHW     
        return image_features, y_true

    def epoch(self):

        loss_list = []
        correct_list = []
        for batch in tqdm(self.train_loader):
            # batch: dict of NTCHW, except for labels
            
            x, y_true = self.prepare_data(batch)  # x: list of N,T,C,H,W

            if self.model is None:
                num_resnet = len(x)
                print(f'Construct {num_resnet} resnet channels')
                num_in_channel_list = [data.size()[2] for data in x]
                self._lazy_init_model(num_in_channel_list)
            self.model.train()
            y_pred = self.model(x)

            loss_tensor = self.loss(y_pred, y_true)
            self.optim.zero_grad()
            loss_tensor.backward()
            self.optim.step()

            # if self.num_step % 100 == 0:
            #     print(f'Step {self.num_step}, loss: {round(loss_tensor.item(), 3)}')
            self.num_step = self.num_step + 1 
            loss_list.append(loss_tensor.item())

            # Compute train correctness
            with torch.no_grad():
                y_pred = torch.argmax(y_pred, dim=-1)
                correct = y_pred == y_true
                correct_list.append(correct)

        loss_avg = np.array(loss_list).mean()
        print(f'loss_avg: {round(loss_avg, 3)}')

        c = torch.concat(correct_list, dim=0)
        accuracy = c.sum() / len(c)
        print(f'Train Accuracy: {round(accuracy.item(), 2)}. ({c.sum().item()} / {len(c)})')
        
    
    def train(self):
        
        for epoch in range(100):
            print(f'Epoch {epoch}')
            self.num_step = 0
            self.epoch()

            # acc = self.valid()
            
            # if acc > self.max_historical_acc:
            #     self.max_historical_acc = acc
            #     self.save_ckpt(epoch, acc)
            
            if (epoch) % 10 == 0:
                acc = self.test()
                if acc > self.max_historical_acc:
                    self.max_historical_acc = acc
                    self.save_ckpt(epoch, acc)

    def valid(self):
        print("Validating ...")
        correct_list = []
        for batch in tqdm(self.valid_loader):
            x, y_true = self.prepare_data(batch)
            with torch.no_grad():
                self.model.eval()
                y_pred = self.model(x)  # N,class_score
                y_pred = torch.argmax(y_pred, dim=-1)
                correct = y_pred == y_true
                correct_list.append(correct)
            
        c = torch.concat(correct_list, dim=0)  # Tensor of prediction correctness
        accuracy = c.sum() / len(c)
        print(f'Eval Accuracy: {round(accuracy.item(), 2)}. ({c.sum().item()} / {len(c)})')
        return accuracy.item()
    
    def test(self):
        print("Testing ...")
        correct_list = []
        for batch in tqdm(self.test_loader):
            # batch is a list of uniformed samples from a single video
            softmax_scores = []
            y_true = None  # y_trues for each batch(1 video) are the same
            for batch_1 in batch:

                x, y_true = self.prepare_data(batch_1)
                if self.model is None:
                    num_in_channel_list = [data.size()[2] for data in x]
                    self._lazy_init_model(num_in_channel_list)

                with torch.no_grad():
                    self.model.eval()
                    y_pred = self.model(x)  # N,class_score
                
                softmax_scores.append(y_pred)
                # y_pred = torch.argmax(y_pred, dim=-1)
                # correct = y_pred == y_true
                # correct = correct.cpu().numpy()
                # correct_list.append(correct)
            # Mean over softmax scores (or logit, the input of softmax)
            mean_score = torch.mean(torch.stack(softmax_scores), dim=0)
            pred_class = torch.argmax(mean_score, dim=-1)
            correct = pred_class == y_true
            correct_list.append(correct)
        
        c = torch.concat(correct_list, axis=0)
        accuracy = c.sum() / len(c)
        print(f'Test Accuracy: {round(accuracy.item(), 2)}. ({c.sum().item()} / {len(c)})')
        return accuracy.item()
    
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
    # trainer.test()
