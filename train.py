import enum
from pathlib import Path
from random import random
from typing import Callable
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

from model.my_slowfast import init_my_slowfast
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:20170"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:20170"
import torch.nn as nn

from pytorchvideo.models.resnet import create_bottleneck_block, create_res_stage
from pytorchvideo.models.stem import create_res_basic_stem

from dataset.chalearn_dataset import ChalearnVideoDataset
from model.multiple_resnet import MultipleResnet
from config.crop_cfg import crop_folder_list

torch.multiprocessing.set_sharing_strategy('file_system')  # Solve the "received 0 items of ancdata" error

# To be improved: 1. mask loss when hand image not detected. * color jitter decreases test acc but do not affect train acc? 

class ModelManager():

    def __init__(self, cfg):
        self.cfg = cfg
        model_name = cfg.MODEL.NAME
        if model_name == "res2d":
            self.init_model = self._init_res2d_model
            self.prepare_data = self._prepare_res2d_data
        elif model_name == "res3d":
            self.init_model = self._init_res3d_model
            self.prepare_data = self._prepare_res3d_data
        elif "slowfast" in model_name:
            self.init_model = self._init_slowfast_model
            self.prepare_data = self._prepare_slowfast_data
        else:
            raise NotImplementedError()
    
    def init_model(self):
        raise NotImplementedError()
    
    def prepare_data(self):
        raise NotImplementedError()

    # # -----------res2d----------------------

    def _init_res2d_model(self):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        model.conv1 = Conv2d(50, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.cuda()
        return model

    def _prepare_res2d_data(self, batch):
        """Prepare data from batch to forward(x)"""
        x = batch[self.cfg.MODEL.R3D_INPUT][:, :, :5].cuda()  # NTCHW. C: bgruv
        N,T,C,H,W = x.size()
        x = torch.reshape(x, (N, T*C, H, W)) 
        y_true = batch['label'].cuda()
        return x, y_true
    
    # # -----------res3d----------------------
    # def _init_res3d_model(self):
    #     model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
    #     model.blocks[0].conv = Conv3d(5, 64, (1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
    #     model.cuda()
    #     return model

    # def _prepare_res3d_data(self, batch):
    #     x = batch[self.cfg.MODEL.R3D_INPUT].cuda()
    #     y_true = batch['label'].cuda()
    #     x = torch.permute(x, [0, 2, 1, 3, 4])  # NTCHW -> NCTHW
    #     return x, y_true
    
    # ----------slow_fast------------------

    def delete_mismatch(self, state_dict):
        layers = [
            'blocks.0.multipathway_blocks.0.conv.weight',
            'blocks.0.multipathway_blocks.1.conv.weight',
            'blocks.6.proj.weight',
            'blocks.6.proj.bias',
            # 'blocks.1.multipathway_blocks.0.res_blocks.0.branch1_conv.weight',
            # 'blocks.1.multipathway_blocks.0.res_blocks.0.branch2.conv_a.weight',
            # 'blocks.2.multipathway_blocks.0.res_blocks.0.branch1_conv.weight',
            # 'blocks.2.multipathway_blocks.0.res_blocks.0.branch2.conv_a.weight',
            # 'blocks.3.multipathway_blocks.0.res_blocks.0.branch1_conv.weight',
            # 'blocks.3.multipathway_blocks.0.res_blocks.0.branch2.conv_a.weight',
            # 'blocks.4.multipathway_blocks.0.res_blocks.0.branch1_conv.weight',
            # 'blocks.4.multipathway_blocks.0.res_blocks.0.branch2.conv_a.weight',
        ]
        for key in layers:
            del state_dict[key]
        return state_dict

    def _init_slowfast_model(self):
        model = init_my_slowfast(self.cfg, (5, 15,), (64, 8,))
        
        pretrained = torch.load(Path('pretrained', 'SLOWFAST_8x8_R50.pyth'))
        state_dict = pretrained["model_state"]

        state_dict = self.delete_mismatch(state_dict)

        model.load_state_dict(state_dict, strict=False)
        model.cuda()
        return model
    
    def _prepare_slowfast_data(self, batch):
        # bgr 0:3; uv 3:5; flow 5:20
        x = batch[self.cfg.MODEL.R3D_INPUT].cuda()  # NTCHW

        
        #  C: 5->1, T: 4 -> 20
        # N,T,C,H,W = x_flow.size()
        # flow叠到temporal上会大幅降低准确率76 - 58
        # x_flow = x_flow.reshape(N, T*5, 3, H, W)  # NTCHW
        # x_flow = torch.permute(x_flow, [0, 2, 1, 3, 4])  # NTCHW -> NCTHW  plt.imshow(x_flow.cpu()[0,:,4].permute((1,2,0))[:,:,:])

        x = torch.permute(x, [0, 2, 1, 3, 4])  # NTCHW -> NCTHW
        # x_bgr = x[:, 0:3]   # plt.imshow(x_bgr.cpu()[0,:,0].permute((1,2,0)))
        # x_uv = x[:, 3:5]    # plt.imshow(x_uv.cpu()[0,:,0].permute((1,2,0))[:,:,1:])
        x_bgruv = x[:, 0:5]
        x_flow = x[:, 5:20]

        # x_depth = x[:, 8:9] # plt.imshow(x_depth.cpu()[0,:,0].permute((1,2,0)))

        y_true = batch['label'].cuda()
        return [x_bgruv, x_flow], y_true  # x_uv, x_flow,  

class Trainer():

    def __init__(self, cfg):
        self.debug = cfg.DEBUG

        if self.debug == True:
            self.num_workers = 0
            self.save_debug_img = True
            
        elif self.debug == False:
            self.num_workers = min(cfg.NUM_CPU, 10)
            self.save_debug_img = False
        
        self.cfg = cfg
        self.batch_size = cfg.CHALEARN.BATCH_SIZE
            
        self.train_dataset = ChalearnVideoDataset(cfg, 'train')
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=cfg.CHALEARN.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=self.num_workers)

        # self.valid_dataset = ChalearnVideoDataset(cfg, 'valid')
        # self.valid_loader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=cfg.CHALEARN.BATCH_SIZE, shuffle=False, drop_last=True, num_workers=self.num_workers)

        self.test_dataset = ChalearnVideoDataset(cfg, 'test')
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=cfg.CHALEARN.BATCH_SIZE, shuffle=False, drop_last=False, num_workers=self.num_workers, collate_fn=lambda x:x)

        self.mm = ModelManager(cfg)
        self.model = self.mm.init_model()
        self.loss = CrossEntropyLoss()
        
        self.num_step = 0
        self.ckpt_dir = Path(cfg.CHALEARN.ROOT, cfg.MODEL.LOGS, cfg.MODEL.CKPT_DIR, cfg.MODEL.NAME)
        self.num_class = cfg.CHALEARN.SAMPLE_CLASS
        self.max_historical_acc = 0.

        self.load_ckpt()

        self.optim = optim.Adam(self.model.parameters(), lr=cfg.MODEL.LR)
        

    def save_ckpt(self, epoch=0, acc=0.0):
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_name = 'acc%.3f_e%d.ckpt' % (acc, epoch)
        # ckpt_name = f'acc{round(acc, 2)}_e{epoch}.ckpt'
        ckpt_path = Path(self.ckpt_dir, ckpt_name)
        
        if not self.debug:
            torch.save(self.model.state_dict(), ckpt_path)
            print(f"Checkpoint saved in {str(ckpt_path)}")
        else:
            print(f'Ignore checkpoint saving under debug mode. {str(ckpt_path)}')
        

    def load_ckpt(self):
        ckpt_list = sorted(glob.glob(str(self.ckpt_dir / '*.ckpt')))
        if len(ckpt_list) == 0:
            print('warning: no checkpoint found, try using HTAH ckeckpoint')
            # Use HTAH checkpoint:
            ckpt_list = sorted(glob.glob(str(Path(self.ckpt_dir.parent, 'slowfast-HTAH', '*.ckpt'))))
            if len(ckpt_list) == 0:
                print('warning: no HTAH checkpoint found')
                return
        ckpt = ckpt_list[-1]
        print(f'loading checkpoint from {str(ckpt)}')
        
        state_dict = torch.load(ckpt)
        # del state_dict['blocks.0.multipathway_blocks.2.conv.weight']
        self.model.load_state_dict(state_dict, strict=True)

        pass

    def train_epoch(self):

        loss_list = []
        correct_list = []
        for batch in tqdm(self.train_loader):
            # batch: dict of NTCHW, except for labels
            
            x, y_true = self.mm.prepare_data(batch)  # x: list of N,T,C,H,W

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
            
            if self.debug:
                break

        loss_avg = np.array(loss_list).mean()
        print(f'loss_avg: {round(loss_avg, 3)}')

        c = torch.concat(correct_list, dim=0)
        accuracy = c.sum() / len(c)
        print(f'Train Accuracy: {round(accuracy.item(), 3)}. ({c.sum().item()} / {len(c)})')
        
    
    def train(self):
        
        if not self.debug:
            max_epoch = self.cfg.MODEL.MAX_EPOCH
        else:
            max_epoch = 3

        for epoch in range(max_epoch):
            print(f'========== Training epoch {epoch}')
            self.num_step = 0
            self.train_epoch()

            # acc = self.valid()
            
            # if acc > self.max_historical_acc:
            #     self.max_historical_acc = acc
            #     self.save_ckpt(epoch, acc)
            
            if (epoch) % 1 == 0:
                y = self.run_eval()

                acc = y['acc']
                if acc > self.max_historical_acc:
                    self.max_historical_acc = acc
                    self.save_ckpt(epoch, acc)
                else:
                    print(f"Not saved. Current best acc: %.3f" % (self.max_historical_acc))
                    
        
        self.save_ckpt(epoch, acc)

    # Run dataloader with eval model
    def run_eval(self, dataset_loader: torch.utils.data.DataLoader=None,):

        prepare_data = self.mm.prepare_data
        model = self.model
        if dataset_loader is None:
            dataset_loader = self.test_loader

        pred_score_list = []  # List of (N, class_score)
        true_list = []  # List of (N,)

        batch_collect = []  # Collect a batch
        samples_per_video = []  # [7, 5, 10, ...]

        def test_batch(collect):
            
            x, y_true = prepare_data(collect)

            with torch.no_grad():
                model.eval()
                y_pred = model(x)  # N,class_score
            
            y_pred = y_pred.cpu().numpy()
            y_true = y_true.cpu().numpy()

            pred_score_list.append(y_pred)
            true_list.append(y_true)
        
        for step, batch in enumerate(tqdm(dataset_loader)):  # LNTCHW, N=1, L for list generated from dataset
            # batch: [[TCHW]]
            # How many samples are uniformly generated from the same video. These are aggregated later through majority voting
            [samples_per_video.append(len(b)) for b in batch]  # b: N,TCHW, N is length of a video - clip_len
            [batch_collect.extend(b) for b in batch]  # batch_collect: N,TCHW
            if len(batch_collect) < self.batch_size:
                continue
            
            while len(batch_collect) > self.batch_size:
                # Batch size reached. Run a batch from batch_collect
                batch_full = default_collate(batch_collect[:self.batch_size])
                batch_collect = batch_collect[self.batch_size:]
                
                test_batch(batch_full)
            
            if self.debug == True and step > 5:
                break
        
        # Last batch
        if len(batch_collect) > 0:
            batch_collect = default_collate(batch_collect)
            test_batch(batch_collect)

        pred_score_arr = np.concatenate(pred_score_list, axis=0)  # (N, class_score)
        pred_score_arr = np.exp(pred_score_arr) / np.sum(np.exp(pred_score_arr), axis=1, keepdims=True)  # (N, class_score)
        # pred_arr = np.argmax(pred_score_arr, axis=1)  # (N,)
        true_arr = np.concatenate(true_list, axis=0)  # (N,)

        # Acc 

        correct_list = []
        read_index = 0
        for num_samples in samples_per_video:
            v_begin = read_index
            v_end = read_index + num_samples
            read_index = read_index + num_samples
            preds = pred_score_arr[v_begin: v_end]  # (N, class_score)
            preds = np.mean(preds, axis=0)  # Mean over N 
            trues = true_arr[v_begin: v_end]
            assert np.all(trues == trues[0])

            # most_pred = np.bincount(preds).argmax()
            pred_1 = np.argmax(preds, axis=0)
            true = trues[0]
            is_correct = (pred_1 == true)

            correct_list.append(is_correct)
            
        c = np.array(correct_list)
        accuracy = c.sum() / len(c)
        print(f'Test Accuracy: {round(accuracy, 3)}. ({c.sum()} / {len(c)})')
        return {
            'ps':pred_score_arr,
            't': true_arr,
            'acc': accuracy,
            'sv': samples_per_video,
        }
    
    def debug_show(self, input):
        debug_folder = Path('logs', 'debug')
        debug_folder.mkdir(parents=True, exist_ok=True)
        # NTCHW
        for frame in range(0, train_cfg.CHALEARN.CLIP_LEN):
            x = input[0][frame].cpu().numpy()
            x = np.transpose(x, (1,2,0))  # HWC, C: BGRUV
            U = x[:, :, 3]
            plt.imshow(U)
            plt.savefig(Path(debug_folder, f'S{str(self.num_step).zfill(2)}_T{str(frame).zfill(2)}'))
            plt.close()
            print('image saved')

if __name__ == '__main__':
    train_cfg = get_cfg()
    yaml_list = ['slowfast-HTAH_1', 'slowfast-HTAH_2', 'slowfast-LHandArm', 'slowfast-LHand', 'slowfast-RHandArm', 'slowfast-RHand']
    for yaml_name in yaml_list:
        train_cfg.merge_from_file(Path('config', yaml_name + '.yaml'))
        override = Path('..', 'cfg_override.yaml')
        if(override.is_file()):  # override after loading local yaml
            train_cfg.merge_from_file(override)
        trainer = Trainer(train_cfg)
        trainer.train()
        # trainer.run_eval()
    # train_cfg.merge_from_file(Path('config', 'slowfast-HTAH.yaml'))
    
    # Trainer(train_cfg).train()
