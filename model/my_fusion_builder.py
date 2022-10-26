from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from pytorchvideo.layers.utils import set_attributes
from pytorchvideo.models.head import create_res_basic_head, create_res_roi_pooling_head
from pytorchvideo.models.net import DetectionBBoxNetwork, MultiPathWayWithFuse, Net
from pytorchvideo.models.resnet import create_bottleneck_block, create_res_stage
from pytorchvideo.models.stem import create_res_basic_stem

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

        conv_dim_in = fusion_dim_in // self.slowfast_channel_reduction_ratio
        conv_fast_to_slow = nn.Conv3d(
            conv_dim_in,
            int(conv_dim_in * self.conv_fusion_channel_ratio),
            kernel_size=self.conv_kernel_size,
            stride=self.conv_stride,
            padding=[k_size // 2 for k_size in self.conv_kernel_size],
            bias=False,
        )
        norm_module = (
            None
            if self.norm is None
            else self.norm(
                num_features=conv_dim_in * self.conv_fusion_channel_ratio,
                eps=self.norm_eps,
                momentum=self.norm_momentum,
            )
        )
        activation_module = None if self.activation is None else self.activation()
        return FuseFastToSlow(
            conv_fast_to_slow=conv_fast_to_slow,
            norm=norm_module,
            activation=activation_module,
        )
    
    @classmethod
    def build_fusion_builder(cls, ):
        return MyFastToSlowFusionBuilder(
            slowfast_channel_reduction_ratio=8,
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
        conv_fast_to_slow: nn.Module,
        norm: Optional[nn.Module] = None,
        activation: Optional[nn.Module] = None,
    ) -> None:
        """
        Args:
            conv_fast_to_slow (nn.module): convolution to perform fusion.
            norm (nn.module): normalization module.
            activation (torch.nn.modules): activation module.
        """
        super().__init__()
        set_attributes(self, locals())

    def forward(self, x):
        return x  # No fusion
        x_s = x[0]
        x_f1 = x[1]
        x_f2 = x[2]
        
        fuse_l = []
        for x_f in [x_f1, x_f2]:
            fuse = self.conv_fast_to_slow(x_f)
            if self.norm is not None:
                fuse = self.norm(fuse)
            if self.activation is not None:
                fuse = self.activation(fuse)
            fuse_l.append(fuse)
        # x_s_fuse = torch.cat([x_s, fuse], 1)
        x_s_fuse = torch.cat([x_s, fuse_l[0], fuse_l[1]], dim=1)
        return [x_s_fuse, x_f1, x_f2]