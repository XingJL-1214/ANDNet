# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.models import layers
from paddleseg.cvlibs import manager, param_init
from paddleseg.models.layers.layer_libs import SyncBatchNorm
from paddle import ParamAttr, reshape, transpose, concat, split
from paddle.nn import Layer, Conv2D, MaxPool2D, AdaptiveAvgPool2D, BatchNorm, Linear
from paddle.nn.initializer import KaimingNormal
from paddle.nn.functional import swish

BatchNorm2d = paddle.nn.BatchNorm2D
bn_mom = 0.1
def SyncBatchNorm(*args, **kwargs):
    """In cpu environment nn.SyncBatchNorm does not have kernel so use nn.BatchNorm2D instead"""
    if paddle.get_device() == 'cpu' or os.environ.get('PADDLESEG_EXPORT_STAGE'):
        return nn.BatchNorm2D(*args, **kwargs)
    elif paddle.distributed.ParallelEnv().nranks == 1:
        return nn.BatchNorm2D(*args, **kwargs)
    else:
        return nn.SyncBatchNorm(*args, **kwargs)


class ConvBNReLU(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2D(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)

        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self._batch_norm = SyncBatchNorm(out_channels, data_format=data_format)
        self._relu = layers.Activation("relu")

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self._relu(x)
        return x

class ConvBNRelU(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, groups=1):
        super(ConvBNRelU, self).__init__()
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=kernel // 2,
            groups=groups,
            bias_attr=False)
        self.bn = SyncBatchNorm(out_channels, data_format='NCHW')
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out

class ConvBN(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()
        self._conv = nn.Conv2D(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)
        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self._batch_norm = SyncBatchNorm(out_channels, data_format=data_format)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        return x


class ConvReLUPool(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1)
        self._relu = layers.Activation("relu")
        self._max_pool = nn.MaxPool2D(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self._relu(x)
        x = self._max_pool(x)
        return x


class SeparableConvBNReLU(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 pointwise_bias=None,
                 **kwargs):
        super().__init__()
        self.depthwise_conv = ConvBN(
            in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            **kwargs)
        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self.piontwise_conv = ConvBNReLU(
            in_channels,
            out_channels,
            kernel_size=1,
            groups=1,
            data_format=data_format,
            bias_attr=pointwise_bias)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.piontwise_conv(x)
        return x


class DepthwiseConvBN(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()
        self.depthwise_conv = ConvBN(
            in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            **kwargs)

    def forward(self, x):
        x = self.depthwise_conv(x)
        return x


class AuxLayer(nn.Layer):
    """
    The auxiliary layer implementation for auxiliary loss.

    Args:
        in_channels (int): The number of input channels.
        inter_channels (int): The intermediate channels.
        out_channels (int): The number of output channels, and usually it is num_classes.
        dropout_prob (float, optional): The drop rate. Default: 0.1.
    """

    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 dropout_prob=0.1,
                 **kwargs):
        super().__init__()

        self.conv_bn_relu = ConvBNReLU(
            in_channels=in_channels,
            out_channels=inter_channels,
            kernel_size=3,
            padding=1,
            **kwargs)

        self.dropout = nn.Dropout(p=dropout_prob)

        self.conv = nn.Conv2D(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=1)

    def forward(self, x):
        x = self.conv_bn_relu(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x

#  ResNet
class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 is_vd_mode=False,
                 act=None,
                 data_format='NCHW'):
        super(ConvBNLayer, self).__init__()
        if dilation != 1 and kernel_size != 3:
            raise RuntimeError("When the dilation isn't 1," \
                "the kernel_size should be 3.")

        self.is_vd_mode = is_vd_mode
        self._pool2d_avg = nn.AvgPool2D(
            kernel_size=2,
            stride=2,
            padding=0,
            ceil_mode=True,
            data_format=data_format)
        self._conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 \
                if dilation == 1 else dilation,
            dilation=dilation,
            groups=groups,
            bias_attr=False,
            data_format=data_format)

        self._batch_norm = layers.SyncBatchNorm(
            out_channels, data_format=data_format)
        self._act_op = layers.Activation(act=act)

    def forward(self, inputs):
        if self.is_vd_mode:
            inputs = self._pool2d_avg(inputs)
        y = self._conv(inputs)
        y = self._batch_norm(y)
        y = self._act_op(y)

        return y


class BottleneckBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 if_first=False,
                 dilation=1,
                 data_format='NCHW'):
        super(BottleneckBlock, self).__init__()

        self.data_format = data_format
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act='relu',
            data_format=data_format)

        self.dilation = dilation

        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act='relu',
            dilation=dilation,
            data_format=data_format)
        self.conv2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            act=None,
            data_format=data_format)

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                kernel_size=1,
                stride=1,
                is_vd_mode=False if if_first or stride == 1 else True,
                data_format=data_format)

        self.shortcut = shortcut
        # NOTE: Use the wrap layer for quantization training
        self.add = layers.Add()
        self.relu = layers.Activation(act="relu")

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = self.add(short, conv2)
        y = self.relu(y)
        return y


class BasicBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 dilation=1,
                 shortcut=True,
                 if_first=False,
                 data_format='NCHW'):
        super(BasicBlock, self).__init__()
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            act='relu',
            data_format=data_format)
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            dilation=dilation,
            act=None,
            data_format=data_format)

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                is_vd_mode=False if if_first or stride == 1 else True,
                data_format=data_format)

        self.shortcut = shortcut
        self.dilation = dilation
        self.data_format = data_format
        self.add = layers.Add()
        self.relu = layers.Activation(act="relu")

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = self.add(short, conv1)
        y = self.relu(y)

        return y

#  STDC
class ConvBNRelu(nn.Layer):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, groups=1):
        super(ConvBNRelu, self).__init__()
        self.conv = nn.Conv2D(
            in_planes,
            out_planes,
            kernel_size=kernel,
            stride=stride,
            padding=kernel // 2,
            groups=groups,
            bias_attr=False)
        self.bn = SyncBatchNorm(out_planes, data_format='NCHW')
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class AddBottleneck(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(AddBottleneck, self).__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.Sequential(
                nn.Conv2D(
                    in_planes,
                    in_planes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=in_planes,
                    bias_attr=False),
                nn.BatchNorm2D(in_planes),
                nn.Conv2D(
                    in_planes, out_planes, kernel_size=1, bias_attr=False),
                nn.BatchNorm2D(out_planes),
            )
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out = x
        for idx, conv in enumerate(self.conv_list):
            if idx == 0 and self.stride == 2:
                out = self.avd_layer(conv(out))
            else:
                out = conv(out)
            out_list.append(out)
        if self.stride == 2:
            x = self.skip(x)
        return paddle.concat(out_list, axis=1) + x


class CatBottleneck(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(CatBottleneck, self).__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias_attr=False),
                nn.BatchNorm2D(out_planes // 2),
            )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out

# M2
def conv_bn(inp, oup, kernel, stride):
    return nn.Sequential(
        nn.Conv2D(
            in_channels=inp,
            out_channels=oup,
            kernel_size=kernel,
            stride=stride,
            padding=(kernel - 1) // 2,
            bias_attr=False),
        nn.BatchNorm2D(num_features=oup, epsilon=1e-05, momentum=0.1),
        nn.ReLU())


class InvertedResidual(nn.Layer):
    def __init__(self, inp, oup, stride, expand_ratio, dilation=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            nn.Conv2D(
                inp,
                inp * expand_ratio,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias_attr=False),
            nn.BatchNorm2D(
                num_features=inp * expand_ratio, epsilon=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Conv2D(
                inp * expand_ratio,
                inp * expand_ratio,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                groups=inp * expand_ratio,
                bias_attr=False),
            nn.BatchNorm2D(
                num_features=inp * expand_ratio, epsilon=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Conv2D(
                inp * expand_ratio,
                oup,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias_attr=False),
            nn.BatchNorm2D(num_features=oup, epsilon=1e-05, momentum=0.1),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

#  M3
class ConvBNLayer3(nn.Layer):
    def __init__(self,
                 in_c,
                 out_c,
                 filter_size,
                 stride,
                 padding,
                 dilation=1,
                 num_groups=1,
                 if_act=True,
                 act=None):
        super(ConvBNLayer3, self).__init__()
        self.if_act = if_act
        self.act = act

        self.conv = nn.Conv2D(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=num_groups,
            bias_attr=False)
        self.bn = layers.SyncBatchNorm(
            num_features=out_c,
            weight_attr=paddle.ParamAttr(
                regularizer=paddle.regularizer.L2Decay(0.0)),
            bias_attr=paddle.ParamAttr(
                regularizer=paddle.regularizer.L2Decay(0.0)))
        self._act_op = layers.Activation(act='hardswish')

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            x = self._act_op(x)
        return x


class ResidualUnit(nn.Layer):
    def __init__(self,
                 in_c,
                 mid_c,
                 out_c,
                 filter_size,
                 stride,
                 use_se,
                 dilation=1,
                 act=None,
                 name=''):
        super(ResidualUnit, self).__init__()
        self.if_shortcut = stride == 1 and in_c == out_c
        self.if_se = use_se

        self.expand_conv = ConvBNLayer3(
            in_c=in_c,
            out_c=mid_c,
            filter_size=1,
            stride=1,
            padding=0,
            if_act=True,
            act=act)

        self.bottleneck_conv = ConvBNLayer3(
            in_c=mid_c,
            out_c=mid_c,
            filter_size=filter_size,
            stride=stride,
            padding='same',
            dilation=dilation,
            num_groups=mid_c,
            if_act=True,
            act=act)
        if self.if_se:
            self.mid_se = SEModule(mid_c, name=name + "_se")
        self.linear_conv = ConvBNLayer3(
            in_c=mid_c,
            out_c=out_c,
            filter_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None)
        self.dilation = dilation

    def forward(self, inputs):
        x = self.expand_conv(inputs)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x = inputs + x
        return x


class SEModule(nn.Layer):
    def __init__(self, channel, reduction=4, name=""):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv1 = nn.Conv2D(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.conv2 = nn.Conv2D(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = F.hardsigmoid(outputs)
        return paddle.multiply(x=inputs, y=outputs)

# Shuffle
def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.shape[0:4]
    channels_per_group = num_channels // groups

    # reshape
    x = reshape(
        x=x, shape=[batch_size, groups, channels_per_group, height, width])

    # transpose
    x = transpose(x=x, perm=[0, 2, 1, 3, 4])

    # flatten
    x = reshape(x=x, shape=[batch_size, num_channels, height, width])
    return x


class ConvBNLayer_shu(nn.Layer):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=1,
            act=None,
            name=None, ):
        super(ConvBNLayer_shu, self).__init__()
        self._conv = Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=ParamAttr(
                initializer=KaimingNormal(), name=name + "_weights"),
            bias_attr=False)

        self._batch_norm = BatchNorm(
            out_channels,
            param_attr=ParamAttr(name=name + "_bn_scale"),
            bias_attr=ParamAttr(name=name + "_bn_offset"),
            act=act,
            moving_mean_name=name + "_bn_mean",
            moving_variance_name=name + "_bn_variance")

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y

class InvertedResidual_shu(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 act="relu",
                 name=None):
        super(InvertedResidual_shu, self).__init__()
        self._conv_pw = ConvBNLayer_shu(
            in_channels=in_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act,
            name='stage_' + name + '_conv1')
        self._conv_dw = ConvBNLayer_shu(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=out_channels // 2,
            act=None,
            name='stage_' + name + '_conv2')
        self._conv_linear = ConvBNLayer_shu(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act,
            name='stage_' + name + '_conv3')

    def forward(self, inputs):
        x1, x2 = split(
            inputs,
            num_or_sections=[inputs.shape[1] // 2, inputs.shape[1] // 2],
            axis=1)
        x2 = self._conv_pw(x2)
        x2 = self._conv_dw(x2)
        x2 = self._conv_linear(x2)
        out = concat([x1, x2], axis=1)
        return channel_shuffle(out, 4)

class InvertedResidualDS(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 act="relu",
                 name=None):
        super(InvertedResidualDS, self).__init__()

        # branch1
        self._conv_dw_1 = ConvBNLayer_shu(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            act=None,
            name='stage_' + name + '_conv4')
        self._conv_linear_1 = ConvBNLayer_shu(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act,
            name='stage_' + name + '_conv5')
        # branch2
        self._conv_pw_2 = ConvBNLayer_shu(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act,
            name='stage_' + name + '_conv1')
        self._conv_dw_2 = ConvBNLayer_shu(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=out_channels // 2,
            act=None,
            name='stage_' + name + '_conv2')
        self._conv_linear_2 = ConvBNLayer_shu(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act,
            name='stage_' + name + '_conv3')

    def forward(self, inputs):
        x1 = self._conv_dw_1(inputs)
        x1 = self._conv_linear_1(x1)
        x2 = self._conv_pw_2(inputs)
        x2 = self._conv_dw_2(x2)
        x2 = self._conv_linear_2(x2)
        out = concat([x1, x2], axis=1)

        return channel_shuffle(out, 4)

# DualSDRV1
class Conv_BN(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, groups=1):
        super(Conv_BN, self).__init__()
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=kernel // 2,
            groups=groups,
            bias_attr=False)
        self.bn = SyncBatchNorm(out_channels, data_format='NCHW')

    def forward(self, x):
        out = self.bn(self.conv(x))
        return out

class CatConv_down(nn.Layer):
    def __init__(self, in_channels=128, out_channels=256, stride=2, groups=[8, 4, 2, 1]):
        super(CatConv_down, self).__init__()

        self.conv1 = ConvBNRelU(in_channels=in_channels, out_channels=out_channels, kernel=3, stride=stride, groups=groups[0])
        self.conv2 = ConvBNRelU(in_channels=out_channels, out_channels=out_channels//2, kernel=3, groups=groups[1])
        self.conv3 = ConvBNRelU(in_channels=out_channels//2, out_channels=out_channels//4, kernel=3, groups=groups[2])
        self.conv4 = ConvBNRelU(in_channels=out_channels//4, out_channels=out_channels//8, kernel=3, groups=groups[3])
        self.conv5 = Conv_BN(in_channels=(out_channels*15)//8, out_channels=out_channels, kernel=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(paddle.concat([x1, x2, x3, x4], axis=1))
        return x5

class CatConv_shortcut(nn.Layer):
    def __init__(self, in_channels=128, out_channels=256, stride=2, groups=[8, 4, 2, 1]):
        super(CatConv_shortcut, self).__init__()

        self.conv1 = ConvBNRelU(in_channels=in_channels, out_channels=out_channels, kernel=3, stride=stride, groups=groups[0])
        self.conv2 = ConvBNRelU(in_channels=out_channels, out_channels=out_channels//2, kernel=3, groups=groups[1])
        self.conv3 = ConvBNRelU(in_channels=out_channels//2, out_channels=out_channels//4, kernel=3, groups=groups[2])
        self.conv4 = ConvBNRelU(in_channels=out_channels//4, out_channels=out_channels//4, kernel=3, groups=groups[3])
        self.conv5 = Conv_BN(in_channels=out_channels*2, out_channels=out_channels, kernel=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(paddle.concat([x1, x2, x3, x4], axis=1))
        return self.relu(x5+x)
class CatConv0(nn.Layer): # -0-
    def __init__(self, in_channels=128, out_channels=256, stride=2, groups=[8, 4, 2, 1]):
        super(CatConv0, self).__init__()

        self.conv1 = ConvBNRelU(in_channels=in_channels, out_channels=out_channels, kernel=3, stride=stride, groups=groups[0])
        self.conv2 = ConvBNRelU(in_channels=out_channels, out_channels=out_channels//2, kernel=3, groups=groups[1])
        self.conv3 = ConvBNRelU(in_channels=out_channels//2, out_channels=out_channels//4, kernel=3, groups=groups[2])
        self.conv4 = ConvBNRelU(in_channels=out_channels//4, out_channels=out_channels//8, kernel=3, groups=groups[3])
        self.conv5 = Conv_BN(in_channels=(out_channels*15)//8, out_channels=out_channels, kernel=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(paddle.concat([x1, x2, x3, x4], axis=1))
        return x5
class CatConv_0(nn.Layer):
    def __init__(self, in_channels=128, out_channels=128, stride=2, groups=[8, 4, 2, 1]):
        super(CatConv_0, self).__init__()

        self.conv1 = ConvBNRelU(in_channels=in_channels, out_channels=out_channels, kernel=3, stride=stride, groups=groups[0])
        self.conv2 = ConvBNRelU(in_channels=out_channels, out_channels=out_channels//2, kernel=3, groups=groups[1])
        self.conv3 = ConvBNRelU(in_channels=out_channels//2, out_channels=out_channels//4, kernel=3, groups=groups[2])
        self.conv4 = ConvBNRelU(in_channels=out_channels//4, out_channels=out_channels//8, kernel=3, groups=groups[3])
        self.conv5 = Conv_BN(in_channels=(out_channels*15)//8, out_channels=out_channels, kernel=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(paddle.concat([x1, x2, x3, x4], axis=1))
        return x5
class CatConv0p(nn.Layer): # -0-
    def __init__(self, in_channels=128, out_channels=256, stride=2, groups=[8, 4, 2, 1]):
        super(CatConv0p, self).__init__()

        self.conv1 = ConvBNRelU(in_channels=in_channels, out_channels=out_channels, kernel=3, stride=stride, groups=groups[0])
        self.conv2 = ConvBNRelU(in_channels=out_channels, out_channels=out_channels//2, kernel=3, groups=groups[1])
        self.conv3 = ConvBNRelU(in_channels=out_channels//2, out_channels=out_channels//4, kernel=3, groups=groups[2])
        self.conv4 = ConvBNRelU(in_channels=out_channels//4, out_channels=out_channels//8, kernel=3, groups=groups[3])
        self.conv5 = Conv_BN(in_channels=(out_channels*15)//8, out_channels=out_channels, kernel=1)
        self.avgpool = nn.Sequential(nn.AvgPool2D(kernel_size=3, stride=2, padding=1))
        self.conv1x1 = layers.SeparableConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(paddle.concat([x1, x2, x3, x4], axis=1)) + self.conv1x1(self.avgpool(x))
        return x5
class CatConv_0p(nn.Layer):
    def __init__(self, in_channels=128, out_channels=128, stride=1, groups=[8, 4, 2, 1]):
        super(CatConv_0p, self).__init__()

        self.conv1 = ConvBNRelU(in_channels=in_channels, out_channels=out_channels, kernel=3, stride=stride, groups=groups[0])
        self.conv2 = ConvBNRelU(in_channels=out_channels, out_channels=out_channels//2, kernel=3, groups=groups[1])
        self.conv3 = ConvBNRelU(in_channels=out_channels//2, out_channels=out_channels//4, kernel=3, groups=groups[2])
        self.conv4 = ConvBNRelU(in_channels=out_channels//4, out_channels=out_channels//8, kernel=3, groups=groups[3])
        self.conv5 = Conv_BN(in_channels=(out_channels*15)//8, out_channels=out_channels, kernel=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(paddle.concat([x1, x2, x3, x4], axis=1)) + x
        return x5
class CatConv0plus0(nn.Layer): # -0-
    def __init__(self, in_channels=128, out_channels=256, stride=2, groups=[8, 4, 2, 1]):
        super(CatConv0plus0, self).__init__()

        self.conv1 = ConvBNRelU(in_channels=in_channels, out_channels=out_channels, kernel=3, stride=stride, groups=groups[0])
        self.conv2 = ConvBNRelU(in_channels=out_channels, out_channels=out_channels//2, kernel=3, groups=groups[1])
        self.conv3 = ConvBNRelU(in_channels=out_channels//2, out_channels=out_channels//4, kernel=3, groups=groups[2])
        self.conv4 = ConvBNRelU(in_channels=out_channels//4, out_channels=out_channels//4, kernel=3, groups=groups[3])
        self.conv5 = Conv_BN(in_channels=out_channels*2, out_channels=out_channels, kernel=1)
        self.avgpool = nn.Sequential(nn.AvgPool2D(kernel_size=3, stride=2, padding=1))
        self.conv1x1 = layers.SeparableConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        a = self.conv1(x)
        b = self.conv2(a)
        c = self.conv3(b)
        d = self.conv4(c)
        c = paddle.concat([c, (c+d)], axis=1)
        b = paddle.concat([b, (c + b)], axis=1)
        a = paddle.concat([a, (a + b)], axis=1)
        out = self.conv5(a) + self.conv1x1(self.avgpool(x))
        return out
class CatConv_0plus0(nn.Layer):
    def __init__(self, in_channels=128, out_channels=128, stride=1, groups=[8, 4, 2, 1]):
        super(CatConv_0plus0, self).__init__()

        self.conv1 = ConvBNRelU(in_channels=in_channels, out_channels=out_channels, kernel=3, stride=stride, groups=groups[0])
        self.conv2 = ConvBNRelU(in_channels=out_channels, out_channels=out_channels//2, kernel=3, groups=groups[1])
        self.conv3 = ConvBNRelU(in_channels=out_channels//2, out_channels=out_channels//4, kernel=3, groups=groups[2])
        self.conv4 = ConvBNRelU(in_channels=out_channels//4, out_channels=out_channels//4, kernel=3, groups=groups[3])
        self.conv5 = Conv_BN(in_channels=out_channels*2, out_channels=out_channels, kernel=1)

    def forward(self, x):
        a = self.conv1(x)
        b = self.conv2(a)
        c = self.conv3(b)
        d = self.conv4(c)
        c = paddle.concat([c, (c + d)], axis=1)
        b = paddle.concat([b, (c + b)], axis=1)
        a = paddle.concat([a, (a + b)], axis=1)
        out = self.conv5(a) + x
        return out
class Catbneck1(nn.Layer): # -1-
    def __init__(self, in_channels=64, out_channels=128, stride=2, groups=[4, 2, 1]):
        super(Catbneck1, self).__init__()

        self.conv1 = ConvBNRelU(in_channels=in_channels, out_channels=out_channels//2, kernel=3, stride=stride, groups=groups[0])
        self.conv2 = ConvBNRelU(in_channels=out_channels//2, out_channels=out_channels//4, kernel=3, groups=groups[1])
        self.conv3 = ConvBNRelU(in_channels=out_channels//4, out_channels=out_channels//4, kernel=3, groups=groups[2])
        self.conv4 = Conv_BN(in_channels=(out_channels*3)//2, out_channels=out_channels, kernel=1)# CONVBNRELU
        self.avgpool = nn.Sequential(nn.AvgPool2D(kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        out = paddle.concat([self.avgpool(x), x1, x2, x3], axis=1)
        return self.conv4(out)
class Catbneck_1(nn.Layer):
    def __init__(self, in_channels=128, out_channels=128, stride=1, groups=[4, 2, 1]):
        super(Catbneck_1, self).__init__()

        self.conv1 = ConvBNRelU(in_channels=in_channels, out_channels=out_channels//2, kernel=3, stride=stride, groups=groups[0])
        self.conv2 = ConvBNRelU(in_channels=out_channels//2, out_channels=out_channels//4, kernel=3, stride=stride, groups=groups[1])
        self.conv3 = ConvBNRelU(in_channels=out_channels//4, out_channels=out_channels//4, kernel=3, stride=stride, groups=groups[2])
        self.conv4 = Conv_BN(in_channels=out_channels*2, out_channels=out_channels, kernel=1, stride=stride)# CONVBNRELU
        # self.avgpool = nn.Sequential(nn.AvgPool2D(kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        out = self.conv4(paddle.concat([x, x1, x2, x3], axis=1))
        return out

class Catbneck2(nn.Layer):
    def __init__(self, in_channels=64, out_channels=128, stride=2, groups=[4, 2, 1]):
        super(Catbneck2, self).__init__()

        self.conv1 = ConvBNRelU(in_channels=in_channels, out_channels=out_channels//2, kernel=3, stride=stride, groups=groups[0])
        self.conv2 = ConvBNRelU(in_channels=out_channels//2, out_channels=out_channels//4, kernel=3, groups=groups[1])
        self.conv3 = ConvBNRelU(in_channels=out_channels//4, out_channels=out_channels//8, kernel=3, groups=groups[2])
        self.conv4 = ConvBNRelU(in_channels=out_channels // 8, out_channels=out_channels // 8, kernel=3, groups=groups[2])
        self.conv5 = Conv_BN(in_channels=(out_channels*3)//2, out_channels=out_channels, kernel=1)# CONVBNRELU
        self.avgpool = nn.Sequential(nn.AvgPool2D(kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        out = paddle.concat([self.avgpool(x), x1, x2, x3, x4], axis=1)
        return self.conv5(out)

class Catbneck_2(nn.Layer):
    def __init__(self, in_channels=128, out_channels=128, stride=1, groups=[4, 2, 1]):
        super(Catbneck_2, self).__init__()

        self.conv1 = ConvBNRelU(in_channels=in_channels, out_channels=out_channels // 2, kernel=3, stride=stride, groups=groups[0])
        self.conv2 = ConvBNRelU(in_channels=out_channels // 2, out_channels=out_channels // 4, kernel=3,  groups=groups[1])
        self.conv3 = ConvBNRelU(in_channels=out_channels // 4, out_channels=out_channels // 8, kernel=3, groups=groups[2])
        self.conv4 = ConvBNRelU(in_channels=out_channels // 8, out_channels=out_channels // 8, kernel=3, groups=groups[2])
        self.conv5 = Conv_BN(in_channels=out_channels*2, out_channels=out_channels, kernel=1, stride=stride)# CONVBNRELU
        # self.avgpool = nn.Sequential(nn.AvgPool2D(kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        out = self.conv5(paddle.concat([x, x1, x2, x3, x4], axis=1))
        return out
class Catbneck_3(nn.Layer):
    def __init__(self, in_channels=128, out_channels=128, stride=1, groups=[4, 2, 1]):
        super(Catbneck_3, self).__init__()

        self.conv1 = ConvBNRelU(in_channels=in_channels, out_channels=out_channels // 2, kernel=3, stride=stride, groups=groups[0])
        self.conv2 = ConvBNRelU(in_channels=out_channels // 2, out_channels=out_channels // 4, kernel=3,  groups=groups[1])
        self.conv3 = ConvBNRelU(in_channels=out_channels // 4, out_channels=out_channels // 4, kernel=3, groups=groups[2])
        self.conv4 = ConvBNRelU(in_channels=out_channels // 4, out_channels=out_channels // 8, kernel=3, groups=groups[2])
        self.conv5 = Conv_BN(in_channels=(out_channels*17)//8, out_channels=out_channels, kernel=1, stride=stride)# CONVBNRELU
        # self.avgpool = nn.Sequential(nn.AvgPool2D(kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        out = self.conv5(paddle.concat([x, x1, x2, x3, x4], axis=1))
        return out
class Catbneck3(nn.Layer):
    def __init__(self, in_channels=64, out_channels=128, stride=2, groups=[4, 2, 1]):
        super(Catbneck3, self).__init__()

        self.conv1 = ConvBNRelU(in_channels=in_channels, out_channels=out_channels//2, kernel=3, stride=stride, groups=groups[0])
        self.conv2 = ConvBNRelU(in_channels=out_channels//2, out_channels=out_channels//4, kernel=3, groups=groups[1])
        self.conv3 = ConvBNRelU(in_channels=out_channels//4, out_channels=out_channels//4, kernel=3, groups=groups[2])
        self.conv4 = ConvBNRelU(in_channels=out_channels // 4, out_channels=out_channels // 8, kernel=3, groups=groups[2])
        self.conv5 = Conv_BN(in_channels=(out_channels*13)//8, out_channels=out_channels, kernel=1)# CONVBNRELU
        self.avgpool = nn.Sequential(nn.AvgPool2D(kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        out = paddle.concat([self.avgpool(x), x1, x2, x3, x4], axis=1)
        return self.conv5(out)
class Catbneck_4(nn.Layer):
    def __init__(self, in_channels=128, out_channels=128, stride=1, groups=[4, 2, 1]):
        super(Catbneck_4, self).__init__()

        self.conv1 = ConvBNRelU(in_channels=in_channels, out_channels=out_channels // 2, kernel=3, stride=stride, groups=groups[0])
        self.conv2 = ConvBNRelU(in_channels=out_channels // 2, out_channels=out_channels // 4, kernel=3,  groups=groups[1])
        self.conv3 = ConvBNRelU(in_channels=out_channels // 4, out_channels=out_channels // 4, kernel=3, groups=groups[2])
        # self.conv4 = ConvBNRelU(in_channels=out_channels // 4, out_channels=out_channels // 8, kernel=3, groups=groups[2])
        self.conv5 = Conv_BN(in_channels=out_channels*2, out_channels=out_channels, kernel=1, stride=stride)# CONVBNRELU
        # self.avgpool = nn.Sequential(nn.AvgPool2D(kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        out = self.conv5(paddle.concat([x, x1, x2, x3], axis=1))
        return out
class Catbneck4(nn.Layer):
    def __init__(self, in_channels=64, out_channels=128, stride=2, groups=[4, 2, 1]):
        super(Catbneck4, self).__init__()

        self.conv1 = ConvBNRelU(in_channels=in_channels, out_channels=out_channels//2, kernel=3, stride=stride, groups=groups[0])
        self.conv2 = ConvBNRelU(in_channels=out_channels//2, out_channels=out_channels//4, kernel=3, groups=groups[1])
        self.conv3 = ConvBNRelU(in_channels=out_channels//4, out_channels=out_channels//4, kernel=3, groups=groups[2])
        # self.conv4 = ConvBNRelU(in_channels=out_channels // 4, out_channels=out_channels // 8, kernel=3, groups=groups[2])
        self.conv5 = Conv_BN(in_channels=(out_channels*3)//2, out_channels=out_channels, kernel=1)# CONVBNRELU
        self.avgpool = nn.Sequential(nn.AvgPool2D(kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        out = paddle.concat([self.avgpool(x), x1, x2, x3], axis=1)
        return self.conv5(out)
class Catbneckl(nn.Layer):
    def __init__(self, in_channels=64, out_channels=128, stride=2, groups=[8, 4, 2, 1]):
        super(Catbneckl, self).__init__()

        self.conv1 = ConvBNRelU(in_channels=in_channels, out_channels=out_channels//2, kernel=3, stride=stride, groups=groups[0])
        self.conv2 = ConvBNRelU(in_channels=out_channels//2, out_channels=out_channels//4, kernel=3, groups=groups[1])
        self.conv3 = ConvBNRelU(in_channels=out_channels//4, out_channels=out_channels//4, kernel=5, groups=groups[2])
        self.conv3 = ConvBNRelU(in_channels=out_channels // 4, out_channels=out_channels // 4, kernel=5,groups=groups[2])
        self.conv4 = Conv_BN(in_channels=(out_channels*3)//2, out_channels=out_channels, kernel=1)# CONVBNRELU
        self.avgpool = nn.Sequential(nn.AvgPool2D(kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        out = paddle.concat([self.avgpool(x), x1, x2, x3], axis=1)
        return self.conv4(out)

class CatBottleneck1_(nn.Layer):
    def __init__(self, in_channels=128, out_channels=128, stride=1, groups=[4, 2, 1]):
        super(CatBottleneck1_, self).__init__()

        self.conv1 = ConvBNRelU(in_channels=in_channels, out_channels=out_channels//2, kernel=1, stride=stride, groups=groups[0])
        self.conv2 = ConvBNRelU(in_channels=out_channels//2, out_channels=out_channels//4, kernel=3, groups=groups[1])
        self.conv3 = ConvBNRelU(in_channels=out_channels//4, out_channels=out_channels//8, kernel=3, groups=groups[2])
        self.conv4 = ConvBNRelU(in_channels=out_channels//8, out_channels=out_channels//8, kernel=3, groups=groups[2])
        # self.conv5 = Conv_BN(in_channels=out_channels*2, out_channels=out_channels, kernel=1)
        self.conv1x1 = layers.SeparableConvBN(in_channels=in_channels*2, out_channels=out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        out = paddle.concat([x1, x2, x3, x4], axis=1) + x
        # out = self.relu(self.conv1x1(paddle.concat([x, out], axis=1)))
        return out
class CatBottleneck1(nn.Layer):
    def __init__(self, in_channels=64, out_channels=128, stride=2, groups=[4, 2, 1]):
        super(CatBottleneck1, self).__init__()

        self.conv1 = ConvBNRelU(in_channels=in_channels, out_channels=out_channels//2, kernel=1, stride=1, groups=groups[0])
        self.conv2 = ConvBNRelU(in_channels=out_channels//2, out_channels=out_channels//4, kernel=3, stride=stride, groups=groups[1])
        self.conv3 = ConvBNRelU(in_channels=out_channels//4, out_channels=out_channels//8, kernel=3, stride=1,groups=groups[2])
        self.conv4 = ConvBNRelU(in_channels=out_channels//8, out_channels=out_channels//8, kernel=3, stride=1,groups=groups[2])
        # self.conv5 = Conv_BN(in_channels=out_channels*2, out_channels=out_channels, kernel=1)
        self.avgpool = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
        self.conv1x1 = layers.SeparableConvBNReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.avgpool(x0)
        x2 = self.conv2(x0)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        out = paddle.concat([x1, x2, x3, x4], axis=1) + self.conv1x1(self.avgpool(x))
        # out = self.relu(self.conv1x1(paddle.concat([x, out], axis=1)))
        return out

class DAPPM(nn.Layer):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()
        self.scale1 = nn.Sequential(nn.AvgPool2D(kernel_size=5, stride=2, padding=2),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2D(inplanes, branch_planes, kernel_size=1, bias_attr=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2D(kernel_size=9, stride=4, padding=4),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2D(inplanes, branch_planes, kernel_size=1, bias_attr=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2D(kernel_size=17, stride=8, padding=8),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2D(inplanes, branch_planes, kernel_size=1, bias_attr=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2D((1, 1)),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2D(inplanes, branch_planes, kernel_size=1, bias_attr=False),
                                    )
        self.scale0 = nn.Sequential(
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(inplanes, branch_planes, kernel_size=1, bias_attr=False),
        )
        self.process1 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(branch_planes, branch_planes, kernel_size=3, padding=1, bias_attr=False),
        )
        self.process2 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(branch_planes, branch_planes, kernel_size=3, padding=1, bias_attr=False),
        )
        self.process3 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(branch_planes, branch_planes, kernel_size=3, padding=1, bias_attr=False),
        )
        self.process4 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(branch_planes, branch_planes, kernel_size=3, padding=1, bias_attr=False),
        )
        self.compression = nn.Sequential(
            BatchNorm2d(branch_planes * 5, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(branch_planes * 5, outplanes, kernel_size=1, bias_attr=False),
        )
        self.shortcut = nn.Sequential(
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(inplanes, outplanes, kernel_size=1, bias_attr=False),
        )

    def forward(self, x):
        # x = self.downsample(x)
        width = x.shape[-1]
        height = x.shape[-2]
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                                                   size=[height, width],
                                                   mode='bilinear') + x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                                                    size=[height, width],
                                                    mode='bilinear') + x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                                                   size=[height, width],
                                                   mode='bilinear') + x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                                                   size=[height, width],
                                                   mode='bilinear') + x_list[3])))

        out = self.compression(paddle.concat(x_list, 1)) + self.shortcut(x)
        return out


