import torch.nn as nn
import logging
import torch.nn.functional as F
from hope.modules.segformer_head import SegFormerHead
from hope.modules.fnc_head import FCNHead
import os
from hope.modules.mixin_vision_transformer import MixVisionTransformer
from functools import partial

logger = logging.getLogger(__name__)

class mit_b5(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class SegFormer(nn.Module):
    def __init__(self, num_classes, load_imagenet_model, imagenet_ckpt_fpath, **kwargs):
        super(SegFormer, self).__init__(**kwargs)

        self.encoder = mit_b5()
        self.head = SegFormerHead(num_classes=num_classes,
                                  in_channels=[64, 128, 320, 512],
                                  channels=128,
                                  in_index=[0,1,2,3],
                                  feature_strides=[4, 8, 16, 32],
                                  #decoder_params=dict(embed_dim=768),
                                  dropout_ratio=0.1,
                                  norm_cfg=dict(type='SyncBN', requires_grad=True),
                                  align_corners=False)
        self.auxi_net = FCNHead(num_convs=1,
                                kernel_size=3,
                                concat_input=True,
                                in_channels=320,
                                num_classes=num_classes,
                                norm_cfg=dict(type='SyncBN', requires_grad=True))
        self.init_weights(load_imagenet_model, imagenet_ckpt_fpath)

    def init_weights(self, load_imagenet_model: bool=False, imagenet_ckpt_fpath: str='') -> None:
        """ For training, we use a models pretrained on ImageNet. Irrelevant at inference.
            Args:
            -   pretrained_fpath: str representing path to pretrained models
            Returns:
            -   None
        """
        logger.info('=> init weights from normal distribution')
        if not load_imagenet_model:
            return
        if os.path.isfile(imagenet_ckpt_fpath):
            print('===========> loading pretrained models {}'.format(imagenet_ckpt_fpath))
            self.encoder.init_weights(pretrained=imagenet_ckpt_fpath)
        else:
            # logger.info(pretrained)
            print('cannot find ImageNet models path, use random initialization')
            raise RuntimeError('no pretrained models found at {}'.format(imagenet_ckpt_fpath))

    def forward(self, inputs):
        h = inputs.size()[2]
        w = inputs.size()[3]
        x = self.encoder(inputs)
        #out = self.head([x[3], x[0]])
        out = self.head(x)
        auxi_out = self.auxi_net(x)
        high_out = F.interpolate(out, size=(h,w), mode='bilinear', align_corners=True)
        return high_out, out, auxi_out

class SegModel(nn.Module):
    def __init__(self, criterions, num_classes, load_imagenet_model, imagenet_ckpt_fpath, **kwargs):
        super(SegModel, self).__init__(**kwargs)
        self.segmodel = SegFormer(num_classes=num_classes,
                                  load_imagenet_model=load_imagenet_model,
                                  imagenet_ckpt_fpath=imagenet_ckpt_fpath)
        self.criterion = None
    def forward(self, inputs, gt=None, label_space=None, others=None):
        high_reso, low_reso, auxi_out = self.segmodel(inputs)
        return high_reso, None, None