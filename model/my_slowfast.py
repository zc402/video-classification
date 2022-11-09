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
from pytorchvideo.models.slowfast import create_slowfast
from torch.nn import CrossEntropyLoss, Module, Linear, Conv2d
from torch import optim
from torch.nn import CrossEntropyLoss, Module, Linear, Conv2d, Conv3d, Identity
from torch.utils.data.dataloader import default_collate
import os

import torch.nn as nn

from pytorchvideo.models.resnet import create_bottleneck_block, create_res_stage
from pytorchvideo.models.stem import create_res_basic_stem

from dataset.chalearn_dataset import ChalearnVideoDataset
from model.multiple_resnet import MultipleResnet
from config.crop_cfg import crop_folder_list

from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from pytorchvideo.layers.utils import set_attributes
from pytorchvideo.models.head import create_res_basic_head, create_res_roi_pooling_head
from pytorchvideo.models.net import DetectionBBoxNetwork, MultiPathWayWithFuse, Net
from pytorchvideo.models.resnet import create_bottleneck_block, create_res_stage
from pytorchvideo.models.stem import create_res_basic_stem

"""
MySlowfast
"""

def init_my_slowfast(cfg, input_channels, stem_dim_outs):
    """
    input_channels= (3, 3,)
    num_class= 400
    """

    num_c = len(input_channels)
    assert num_c >= 2

    # slowfast_conv_channel_fusion_ratio = 0  # pathway_stage_dim_in = [ stage_dim_in + stage_dim_in * slowfast_conv_channel_fusion_ratio // slowfast_channel_reduction_ratio[0],]
    slow_c = stem_dim_outs[0]
    fast_cs = stem_dim_outs[1:]
    # reduction ratio
    ratio = (np.array(slow_c, np.int) // np.array(fast_cs, np.int))
    ratio = tuple(ratio.tolist())
    slowfast_channel_reduction_ratio = ratio  # (8,8,) for 2 fastways.  # 64(slow) / 8(fast) = 8
    
    assert len(stem_dim_outs) == num_c

    stem_function = tuple([create_res_basic_stem,] * num_c)
    stem_conv_kernel_sizes = tuple([(1, 7, 7) for _ in range(num_c)])
    stem_conv_strides = tuple([(1, 2, 2) for _ in range(num_c)])
    stem_pool = tuple([nn.MaxPool3d for _ in range(num_c)])
    stem_pool_kernel_sizes = tuple([(1, 3, 3) for _ in range(num_c)])
    stem_pool_strides = tuple([(1, 2, 2) for _ in range(num_c)])
    stage_conv_a_kernel_sizes = [((1, 1, 1), (1, 1, 1), (3, 1, 1), (3, 1, 1)), ] + [((3, 1, 1), (3, 1, 1), (3, 1, 1), (3, 1, 1)) for _ in range(num_c - 1)]
    stage_conv_b_kernel_sizes = tuple([((1, 3, 3), (1, 3, 3), (1, 3, 3), (1, 3, 3)) for _ in range(num_c)] )
    stage_conv_b_num_groups = tuple([(1, 1, 1, 1) for _ in range(num_c)])
    stage_conv_b_dilations = tuple([((1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)) for _ in range(num_c)] )
    stage_spatial_strides = tuple([(1, 2, 2, 2) for _ in range(num_c)])
    stage_temporal_strides = tuple([(1, 1, 1, 1) for _ in range(num_c)])
    head_pool_kernel_sizes = tuple([(8, 2, 2) for _ in range(num_c)])

    bottleneck = tuple([
        (
            create_bottleneck_block,
            create_bottleneck_block,
            create_bottleneck_block,
            create_bottleneck_block,
        )
        for _ in range(num_c)])

    fuse = cfg.MODEL.FUSE
    if fuse:
        fusion_builder = MyFastToSlowFusionBuilder.build_fusion_builder(slowfast_channel_reduction_ratio[0]).create_module
        slowfast_conv_channel_fusion_ratio = 2*(num_c - 1)
    else:
        fusion_builder = nn.Identity
        slowfast_conv_channel_fusion_ratio = 0

    model = create_slowfast(
        # SlowFast configs.
        slowfast_channel_reduction_ratio = slowfast_channel_reduction_ratio,  # If slow has 64 channels, fast has 16 channels, then 64/16=4
        slowfast_conv_channel_fusion_ratio = slowfast_conv_channel_fusion_ratio,  # 2*2: 2 fast 
        model_depth=50,
        model_num_class=cfg.CHALEARN.NUM_CLASS,
        input_channels=input_channels,
        fusion_builder = fusion_builder,

        # slowfast_fusion_conv_stride=(1,1,1),
        # slowfast_fusion_conv_kernel_size=(7, 1, 1),

        # Stem configs.
        stem_function = stem_function,
        stem_dim_outs=stem_dim_outs,
        stem_conv_kernel_sizes = stem_conv_kernel_sizes,
        stem_conv_strides = stem_conv_strides,
        stem_pool = stem_pool,
        stem_pool_kernel_sizes = stem_pool_kernel_sizes,
        stem_pool_strides = stem_pool_strides,
        
        # Stage configs.
        stage_conv_a_kernel_sizes = stage_conv_a_kernel_sizes,
        stage_conv_b_kernel_sizes = stage_conv_b_kernel_sizes,
        stage_conv_b_num_groups = stage_conv_b_num_groups,
        stage_conv_b_dilations = stage_conv_b_dilations,
        stage_spatial_strides = stage_spatial_strides,
        stage_temporal_strides = stage_temporal_strides,
        bottleneck = bottleneck,
        # Head configs.
        head_pool_kernel_sizes = head_pool_kernel_sizes,
        )
    return model


_MODEL_STAGE_DEPTH = {
    18: (1, 1, 1, 1),
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
    152: (3, 8, 36, 3),
}

class MyFastToSlowFusionBuilder:
    def __init__(
        self,
        slowfast_channel_reduction_ratio: int,
        conv_fusion_channel_ratio: float,
        conv_kernel_size: Tuple[int],
        conv_stride: Tuple[int],
        norm: Callable = nn.BatchNorm3d,
        norm_eps: float = 1e-5,
        norm_momentum: float = 0.1,
        activation: Callable = nn.ReLU,
        max_stage_idx: int = 3,
    ) -> None:
        """
        Given a list of two tensors from Slow pathway and Fast pathway, fusion information
        from the Fast pathway to the Slow on through a convolution followed by a
        concatenation, then return the fused list of tensors from Slow and Fast pathway in
        order.
        Args:
            slowfast_channel_reduction_ratio (int): Reduction ratio from the stage dimension.
                Used to compute conv_dim_in = fusion_dim_in // slowfast_channel_reduction_ratio
            conv_fusion_channel_ratio (int): channel ratio for the convolution used to fuse
                from Fast pathway to Slow pathway.
            conv_kernel_size (int): kernel size of the convolution used to fuse from Fast
                pathway to Slow pathway.
            conv_stride (int): stride size of the convolution used to fuse from Fast pathway
                to Slow pathway.
            norm (callable): a callable that constructs normalization layer, examples
                include nn.BatchNorm3d, None (not performing normalization).
            norm_eps (float): normalization epsilon.
            norm_momentum (float): normalization momentum.
            activation (callable): a callable that constructs activation layer, examples
                include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not performing
                activation).
            max_stage_idx (int): Returns identity module if we exceed this
        """
        set_attributes(self, locals())

    def create_module(self, fusion_dim_in: int, stage_idx: int) -> nn.Module:
        """
        Creates the module for the given stage
        Args:
            fusion_dim_in (int): input stage dimension
            stage_idx (int): which stage this is
        """
        if stage_idx > self.max_stage_idx:
            return nn.Identity()

        slow_in_channels = fusion_dim_in
        slow_out_channels = slow_in_channels
        fast_in_channels = fusion_dim_in // self.slowfast_channel_reduction_ratio
        fast_out_channels = int(fast_in_channels * self.conv_fusion_channel_ratio)
        num_fast_ways = 1
        fuse_out_channels = slow_out_channels + (fast_out_channels * num_fast_ways)

        # conv_dim_in = fusion_dim_in // self.slowfast_channel_reduction_ratio

        
        conv_fast_to_slow = nn.ModuleList([
            nn.Conv3d(
            fast_in_channels,
            fast_out_channels,
            kernel_size=self.conv_kernel_size,
            stride=self.conv_stride,
            padding=[k_size // 2 for k_size in self.conv_kernel_size],
            bias=False,
        ) for _ in range(num_fast_ways)])

        residual = nn.Sequential(
            nn.Conv3d(
                slow_in_channels, 
                fuse_out_channels,
                kernel_size=(1, 1, 1),
                stride=(1, 1, 1),
                padding=(0, 0, 0),
                bias=True),
            nn.ReLU(inplace=True),
        )

        norm_module = nn.ModuleList([
            None
            if self.norm is None
            else self.norm(
                num_features=fast_in_channels * self.conv_fusion_channel_ratio,
                eps=self.norm_eps,
                momentum=self.norm_momentum,
            ) for _ in range(num_fast_ways)])
        
        activation_module = nn.ModuleList([
            None if self.activation is None else self.activation()
            for _ in range(num_fast_ways)])

        res_unit = nn.Sequential(
            nn.Conv3d(fuse_out_channels, fuse_out_channels, (3, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(fuse_out_channels),
            nn.Conv3d(fuse_out_channels, fuse_out_channels, (1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(fuse_out_channels),
            nn.Conv3d(fuse_out_channels, fuse_out_channels, (3, 1, 1), padding=(1, 0, 0)),
        )

        return FuseFastToSlow(
            conv_fast_to_slow=conv_fast_to_slow,
            residual=residual,
            norm=norm_module,
            activation=activation_module,
            res_unit = res_unit
        )
    
    @classmethod
    def build_fusion_builder(cls, slowfast_channel_reduction_ratio):
        return MyFastToSlowFusionBuilder(
            slowfast_channel_reduction_ratio=slowfast_channel_reduction_ratio,
                conv_fusion_channel_ratio=2,
                conv_kernel_size=(7, 1, 1),
                conv_stride=(1,1,1),
                norm=nn.BatchNorm3d,
                activation=nn.ReLU,
                max_stage_idx=len(_MODEL_STAGE_DEPTH[50]) - 1,
            )


class FuseFastToSlow(nn.Module):
    """
    Given a list of two tensors from Slow pathway and Fast pathway, fusion information
    from the Fast pathway to the Slow on through a convolution followed by a
    concatenation, then return the fused list of tensors from Slow and Fast pathway in
    order.
    """

    def __init__(
        self,
        conv_fast_to_slow: nn.ModuleList,
        residual: nn.Module,
        norm: Optional[nn.ModuleList] = None,
        activation: Optional[nn.ModuleList] = None,
        res_unit: nn.Module = None,
    ) -> None:
        """
        Args:
            conv_fast_to_slow (nn.module): convolution to perform fusion.
            norm (nn.module): normalization module.
            activation (torch.nn.modules): activation module.
        """
        super().__init__()
        set_attributes(self, locals())
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x):
        # return x  # No fusion
        x_s = x[0]  # x slow layer
        x_fs = x[1:]  # x fast layers

        fuse_list = []
        for i, x_f in enumerate(x_fs):
            fuse = self.conv_fast_to_slow[i](x_f)
            if self.norm[i] is not None:
                fuse = self.norm[i](fuse)
            if self.activation[i] is not None:
                fuse = self.activation[i](fuse)
            fuse_list.append(fuse)
        fuse_cat = torch.cat(fuse_list, dim=1)  # NCTHW, fast features to be fused with slow

        x_s_fuse = torch.cat([x_s, fuse_cat], dim=1)
        
        x_s_fuse = self.res_unit(x_s_fuse)

        x_s_residual = self.residual(x_s)

        x_s_fuse = x_s_fuse + x_s_residual
        x_s_fuse = self.relu(x_s_fuse)

        return [x_s_fuse, *x_fs]

    # def forward(self, x):
    #     x_s = x[0]
    #     if len(x[1:]) == 1:
    #         x_f = x[1]
    #     else:
    #         x_f = torch.sum(torch.stack(x[1:]), dim=0)
    #     fuse = self.conv_fast_to_slow(x_f)
    #     if self.norm is not None:
    #         fuse = self.norm(fuse)
    #     if self.activation is not None:
    #         fuse = self.activation(fuse)
    #     x_s_fuse = torch.cat([x_s, fuse], 1)
    #     return [x_s_fuse, *x[1:]]
