import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from .decoder_head import BaseDecodeHead


class FCNHead(nn.Module):
    """Fully Convolution Networks for Semantic Segmentation.
    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.
    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
    """

    def __init__(self,
                 num_convs=1,
                 kernel_size=3,
                 in_channels=320,
                 num_classes=150,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 **kwargs):
        assert num_convs >= 0
        self.kernel_size = kernel_size
        super(FCNHead, self).__init__()
        self.in_channels = in_channels
        self.channels = 256
        self.num_classes = num_classes
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,  #
                conv_cfg=None,
                norm_cfg=self.norm_cfg, # sync bn
                act_cfg=self.act_cfg)) # relu

        self.convs = nn.Sequential(*convs)
        self.cls_seg = nn.Conv2d(in_channels=self.channels, out_channels=self.num_classes,
                                 kernel_size=1)

    def forward(self, inputs):
        """Forward function."""
        x = inputs[-2]
        output = self.convs(x)
        output = self.cls_seg(output)
        return output