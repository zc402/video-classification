import enum
from pathlib import Path
from random import random
from turtle import forward
from typing import Callable, Dict, List
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
from pytorchvideo.models.slowfast import create_slowfast
from torch.nn import CrossEntropyLoss, Module, Linear, Conv2d
from torch import optim
from torch.nn import CrossEntropyLoss, Module, Linear, Conv2d, Conv3d, Identity
from config.defaults import get_cfg, get_override_cfg
from torch.utils.data.dataloader import default_collate
import os
from dataset.chalearn_dataset import ChalearnVideoDataset
from train import Trainer, ModelManager
import pickle
import torch.nn as nn

class ResultSaver:
    # Save evaluation results (including last dense layer) into file system
    

    def load_part_cfgs(self):
        # Load configs for each part
        cfg = get_cfg()
        yaml_list = ['slowfast-HTAH', 'slowfast-LHandArm', 'slowfast-LHand', 'slowfast-RHandArm', 'slowfast-RHand']
        for yaml_name in yaml_list:
            cfg.merge_from_file(Path('config', yaml_name + '.yaml'))
            override = Path('..', 'cfg_override.yaml')
            if override.is_file():  # override after loading local yaml
                cfg.merge_from_file(override)
            yield cfg
    
    def save_network_output(self):
        # Load config
        for cfg in self.load_part_cfgs():
        
            # Construct data loaders with NO shuffle
            debug = cfg.DEBUG

            if debug == True:
                self.num_workers = 0
            elif debug == False:
                self.num_workers = min(cfg.NUM_CPU, 10)

            sampling = 'uniform'
            self.train_dataset = ChalearnVideoDataset(cfg, 'train', sampling)
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=cfg.CHALEARN.BATCH_SIZE, shuffle=False, drop_last=False, num_workers=self.num_workers, collate_fn=lambda x:x)

            # self.valid_dataset = ChalearnVideoDataset(cfg, 'valid', sampling)
            # self.valid_loader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=cfg.CHALEARN.BATCH_SIZE, shuffle=False, drop_last=False, num_workers=self.num_workers, collate_fn=lambda x:x)

            self.test_dataset = ChalearnVideoDataset(cfg, 'test', sampling)
            self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=cfg.CHALEARN.BATCH_SIZE, shuffle=False, drop_last=False, num_workers=self.num_workers, collate_fn=lambda x:x)
            
            def save_eval_materials(name_of_set):
                # Save eval result into file system
                eval_save_path = Path(cfg.CHALEARN.ROOT, cfg.MODEL.LOGS, 'sparse_fusion', name_of_set, cfg.MODEL.NAME)
                if name_of_set == 'train':
                    loader = self.train_loader
                elif name_of_set == 'test':
                    loader = self.test_loader
                else:
                    raise NotImplementedError()

                trainer = Trainer(cfg)

                y = trainer.run_eval(loader)

                print(f"eval acc {y['acc']}")

                eval_save_path.parent.mkdir(parents=True, exist_ok=True)
                with eval_save_path.open('wb') as f:
                    pickle.dump(y, f)
            
            # save_eval_materials('train')
            save_eval_materials('test')

class SparseModel(nn.Module):

    def __init__(self, num_class, num_part, ) -> None:
        super().__init__()
        self.num_class = num_class
        self.num_part = num_part
        self.fcs = nn.ModuleList([nn.Linear(num_part, 1) for _ in range(num_class)])
    
    def forward(self, x):
        # x.shape: (N, P, C)
        assert x.size()[1:3] == (self.num_part, self.num_class)
        y = []
        for c in range(self.num_class):
            yc = self.fcs[c](x[:, :, c])  # (N, 1)
            y.append(yc)
        y = torch.concat(y, dim=-1)  # (N,C)
        return y

class SparseFusionDataset(torch.utils.data.Dataset):

    def __init__(self, res_folder: Path) -> None:
        super().__init__()
        
        self.part_res_list = []  # [filename, {p,t,ps: arr}]
        res_paths = glob.glob(str(Path(res_folder, '*')))
        for p in res_paths:
            with Path(p).open('rb') as f:
                res_dict = pickle.load(f)
                name = Path(p).stem
                self.part_res_list.append((name, res_dict))

        self.part_res_list = sorted(self.part_res_list, key=lambda x: x[0])  # sort by part name

        self.T_cp = [part[1]['t'] for part in self.part_res_list]  # ground truth (part, sample)
        self.T_cp = np.stack(self.T_cp)
        self.T_cp = self.T_cp[0, :]  # ground truth for each part should be same

        self.PS_cp = [part[1]['ps'] for part in self.part_res_list]  # scores (part, sample, class)
        self.PS_cp = np.stack(self.PS_cp)

        self.sv = [part[1]['sv'] for part in self.part_res_list]
        self.sv = np.stack(self.sv)
        self.sv = self.sv[0, :]

        self.num_part, self.num_N, self.num_class = self.PS_cp.shape
        
    def __len__(self):
        return self.T_cp.shape[0]  # sample
    
    def __getitem__(self, index):


        
        return {
            't': self.T_cp[index],
            'ps': self.PS_cp[:, index],
        }
        # Output: t - (sample, part); ps - (sample, part, class)
            
                

class SparseTrainer:

    def __init__(self) -> None:
        cfg = get_override_cfg()
        train_material_save_folder = Path(cfg.CHALEARN.ROOT, cfg.MODEL.LOGS, 'sparse_fusion', 'train')
        self.train_dataset = SparseFusionDataset(train_material_save_folder)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=200, shuffle=True)

        test_material_save_folder = Path(cfg.CHALEARN.ROOT, cfg.MODEL.LOGS, 'sparse_fusion', 'test')
        self.test_dataset = SparseFusionDataset(test_material_save_folder)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=200)

        self.sparse_model = SparseModel(self.train_dataset.num_class, self.train_dataset.num_part).cuda()
        self.optim = optim.Adam(self.sparse_model.parameters(), lr=1e-3)
        self.loss = CrossEntropyLoss()

        self.max_accuracy = 0.
        self.ckpt_folder = Path(cfg.CHALEARN.ROOT, cfg.MODEL.LOGS, 'sparse_fusion_ckpt')

    def train(self):
        
        for epoch in range(2000):
            # print(f'Running epoch {epoch}')
            for step, batch in enumerate(self.train_loader):
                self.sparse_model.train()
                P_nc = self.sparse_model(batch['ps'].cuda())  # predicted score (N,P,C)
                T_n = batch['t'].cuda()  # ground truth

                loss_tensor = self.loss(P_nc, T_n)
                self.optim.zero_grad()
                loss_tensor.backward()
                self.optim.step()

                if step % 100 == 0:
                    pass
                    # print(loss_tensor.item())
                    # correctness = (T_n.detach().cpu().numpy() == np.argmax(P_nc.detach().cpu().numpy(), axis=1))
                    # print(np.mean(correctness))

            if (epoch+1) % 20 == 0:
                self.test(epoch)
            if (epoch+1) % 100 == 0:
                print("Epoch:%d" % epoch)
        
    def save_ckpt(self, acc, epoch):
        self.ckpt_folder.mkdir(parents=True, exist_ok=True)
        ckpt_path = Path(self.ckpt_folder, 'acc-%.3f-epoch-%d' % (acc, epoch))
        torch.save(self.sparse_model.state_dict(), ckpt_path)

    def test(self, epoch=0):
        # correct_samples = []  # whether a sample prediction is correct

        pred_score_list = []  # (L, N, C,) pred_score for each sample
        true_list = []

        for step, batch in enumerate(self.test_loader):
            with torch.no_grad():
                self.sparse_model.eval()
                P_nc = self.sparse_model(batch['ps'].cuda())  # predicted score (N,P,C)
                T_n = batch['t'].cuda()  # ground truth

                pred_score_list.append(P_nc.cpu().numpy())
                true_list.append(T_n.cpu().numpy())
        
        pred_score_arr = np.concatenate(pred_score_list, axis=0)  # (N,C,)
        # pred_arr = np.argmax(pred_score_arr, axis=1)
        true_arr = np.concatenate(true_list, axis=0)  # (N)

        # Whether a video prediction is correct
        correct_list = []
        read_index = 0
        samples_per_video = self.test_dataset.sv
        for num_samples in samples_per_video:
            v_begin = read_index
            v_end = read_index + num_samples
            read_index = read_index + num_samples
            preds = pred_score_arr[v_begin: v_end]  # N,C
            preds = np.mean(preds, axis=0)  # C
            trues = true_arr[v_begin: v_end]
            assert np.all(trues == trues[0])

            pred_1 = np.argmax(preds, axis=0)
            true = trues[0]
            is_correct = (pred_1 == true)

            correct_list.append(is_correct)
        
        accuracy = np.mean(correct_list)
        
        if accuracy > self.max_accuracy:
            self.save_ckpt(accuracy, epoch)
        self.max_accuracy = max(accuracy, self.max_accuracy)

        print(f'Max accuracy: %.3f, new test accuracy: %.3f' % (self.max_accuracy, accuracy))

        



if __name__ == '__main__':
    # ResultSaver().save_network_output()
    SparseTrainer().train()
    # SparseTrainer().test()