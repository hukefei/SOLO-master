"""modify
1. Remove squeeze-and-excite (SE)
2. Replace all swish with relu
3. Change same padding
4. Fix the stem(32) and head(1280)
5. Remove dropout

pretrained model from https://github.com/rwightman/pytorch-image-models/releases/tag/v0.1-weights
(tf_efficientnet_lite3-b733e338.pth)
"""

from torch import nn

from .utils import (
    round_filters,
    round_repeats,
    get_model_params,
)

import numpy as np
from mmdet.utils import get_root_logger
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from ..registry import BACKBONES


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=kernel_size // 2, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        return x


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self.conv_pw = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self.conv_dw = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Output phase
        final_oup = self._block_args.output_filters
        self.conv_pwl = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self.relu = nn.ReLU()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self.conv_pw(inputs)
            x = self.bn1(x)
            x = self.relu(x)

        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv_pwl(x)
        x = self.bn3(x)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            x = x + inputs  # skip connection
        return x


@BACKBONES.register_module
class EfficientNet_Lite(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """

    def __init__(self,
                 model_name='efficientnet-b3',
                 num_stages=7,
                 out_indices=(1, 2, 4, 6),
                 frozen_stages=-1,
                 norm_eval=True):
        super().__init__()
        blocks_args, global_params = self.from_name(model_name)
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args[:num_stages]
        self.num_stages = num_stages
        assert 1 <= num_stages <= 7
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        # print(global_params)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = 32  # number of output channels
        self.conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self.blocks = nn.ModuleList([])
        for i, block_args in enumerate(self._blocks_args):
            _blocks = []
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )
            if i == 0:
                block_args = block_args._replace(input_filters=32, num_repeat=1)
            if i == 6:
                block_args = block_args._replace(num_repeat=1)
            # print(block_args)
            # print('-' * 100)
            # The first block needs to take care of stride and filter size increase.
            _blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                _blocks.append(MBConvBlock(block_args, self._global_params))
            self.blocks.append(nn.Sequential(*_blocks))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = 1280
        self.conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self.relu = nn.ReLU()

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = self.relu(self.bn1(self.conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self.blocks):
            x = block(x)

        # Head
        x = self.relu(self.bn2(self.conv_head(x)))

        return x

    def forward(self, inputs):

        # Stem
        x = self.relu(self.bn1(self.conv_stem(inputs)))

        # Blocks
        outs = []
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str and should not be None')

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for m in (self.conv_stem, self.bn1):
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(self.frozen_stages):
            for m in self.blocks[i]:
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(EfficientNet_Lite, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, (nn.BatchNorm2d)):
                    m.eval()

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return blocks_args, global_params

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. """
        valid_models = ['efficientnet-b'+str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))
