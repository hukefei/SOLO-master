"""
code borrowed from https://github.com/stigma0617/VoVNet.pytorch/blob/master/models_vovnet/vovnet.py
"""
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from collections import OrderedDict
from mmdet.utils import get_root_logger
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from ..registry import BACKBONES

__all__ = ['VoVNet', 'vovnet27_slim', 'vovnet39', 'vovnet57']

model_urls = {
    'vovnet39': 'https://dl.dropbox.com/s/1lnzsgnixd8gjra/vovnet39_torchvision.pth?dl=1',
    'vovnet57': 'https://dl.dropbox.com/s/6bfu9gstbwfw31m/vovnet57_torchvision.pth?dl=1'
}


def conv3x3(in_channels, out_channels, module_name, postfix,
            stride=1, groups=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [
        ('{}_{}/conv'.format(module_name, postfix),
         nn.Conv2d(in_channels, out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   groups=groups,
                   bias=False)),
        ('{}_{}/norm'.format(module_name, postfix),
         nn.BatchNorm2d(out_channels)),
        ('{}_{}/relu'.format(module_name, postfix),
         nn.ReLU(inplace=True)),
    ]


def conv1x1(in_channels, out_channels, module_name, postfix,
            stride=1, groups=1, kernel_size=1, padding=0):
    """1x1 convolution"""
    return [
        ('{}_{}/conv'.format(module_name, postfix),
         nn.Conv2d(in_channels, out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   groups=groups,
                   bias=False)),
        ('{}_{}/norm'.format(module_name, postfix),
         nn.BatchNorm2d(out_channels)),
        ('{}_{}/relu'.format(module_name, postfix),
         nn.ReLU(inplace=True)),
    ]


class _OSA_module(nn.Module):
    def __init__(self,
                 in_ch,
                 stage_ch,
                 concat_ch,
                 layer_per_block,
                 module_name,
                 identity=False):
        super(_OSA_module, self).__init__()

        self.identity = identity
        self.layers = nn.ModuleList()
        in_channel = in_ch
        for i in range(layer_per_block):
            self.layers.append(nn.Sequential(
                OrderedDict(conv3x3(in_channel, stage_ch, module_name, i))))
            in_channel = stage_ch

        # feature aggregation
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(
            OrderedDict(conv1x1(in_channel, concat_ch, module_name, 'concat')))

    def forward(self, x):
        identity_feat = x
        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        xt = self.concat(x)

        if self.identity:
            xt = xt + identity_feat

        return xt


class _OSA_stage(nn.Sequential):
    def __init__(self,
                 in_ch,
                 stage_ch,
                 concat_ch,
                 block_per_stage,
                 layer_per_block,
                 stage_num):
        super(_OSA_stage, self).__init__()

        if not stage_num == 2:
            self.add_module('Pooling',
                            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        module_name = f'OSA{stage_num}_1'
        self.add_module(module_name,
                        _OSA_module(in_ch,
                                    stage_ch,
                                    concat_ch,
                                    layer_per_block,
                                    module_name))
        for i in range(block_per_stage - 1):
            module_name = f'OSA{stage_num}_{i+2}'
            self.add_module(module_name,
                            _OSA_module(concat_ch,
                                        stage_ch,
                                        concat_ch,
                                        layer_per_block,
                                        module_name,
                                        identity=True))


@BACKBONES.register_module
class VoVNet(nn.Module):

    config_stage_ch = {
        'vovnet27_slim': [64, 80, 96, 112],
        'vovnet39': [128, 160, 192, 224],
        'vovnet57': [128, 160, 192, 224]
    }
    config_concat_ch = {
        'vovnet27_slim': [128, 256, 384, 512],
        'vovnet39': [256, 512, 768, 1024],
        'vovnet57': [256, 512, 768, 1024]
    }
    block_per_stage = {
        'vovnet27_slim': [1, 1, 1, 1],
        'vovnet39': [1, 1, 2, 2],
        'vovnet57': [1, 1, 4, 3]
    }
    layer_per_block = 5

    def __init__(self,
                 model_name,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 norm_eval=True):
        super(VoVNet, self).__init__()
        self.config_stage_ch = self.config_stage_ch[model_name]
        self.config_concat_ch = self.config_concat_ch[model_name]
        self.block_per_stage = self.block_per_stage[model_name]
        self.layer_per_block = self.layer_per_block
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        # Stem module
        stem = conv3x3(3, 64, 'stem', '1', 2)
        stem += conv3x3(64, 64, 'stem', '2', 1)
        stem += conv3x3(64, 128, 'stem', '3', 2)
        self.add_module('stem', nn.Sequential(OrderedDict(stem)))

        stem_out_ch = [128]
        in_ch_list = stem_out_ch + self.config_concat_ch[:-1]
        self.stage_names = []
        for i in range(4):
            name = 'stage%d' % (i + 2)
            self.stage_names.append(name)
            self.add_module(name,
                            _OSA_stage(in_ch_list[i],
                                       self.config_stage_ch[i],
                                       self.config_concat_ch[i],
                                       self.block_per_stage[i],
                                       self.layer_per_block,
                                       i + 2))

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str and should not be None')

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            m = getattr(self, 'stem')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'stage{}'.format(i + 1))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = getattr(self, 'stem')(x)
        outs = []
        for i, name in enumerate(self.stage_names):
            x = getattr(self, name)(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


def _vovnet(arch,
            pretrained,
            progress,
            **kwargs):
    model = VoVNet(arch,**kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def vovnet57(pretrained=False, progress=True, **kwargs):
    r"""Constructs a VoVNet-57 model as described in
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vovnet('vovnet57', pretrained, progress, **kwargs)


def vovnet39(pretrained=False, progress=True, **kwargs):
    r"""Constructs a VoVNet-39 model as described in
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vovnet('vovnet39', pretrained, progress, **kwargs)


def vovnet27_slim(pretrained=False, progress=True, **kwargs):
    r"""Constructs a VoVNet-39 model as described in
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vovnet('vovnet27_slim', pretrained, progress, **kwargs)
