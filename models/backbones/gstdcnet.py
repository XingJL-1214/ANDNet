# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import math

import paddle
import paddle.nn as nn
from paddleseg.models import layers
from paddleseg.utils import utils
from paddleseg.cvlibs import manager, param_init
from paddleseg.models.layers.layer_libs import SyncBatchNorm

__all__ = ["GSTDC1", "GSTDC2"]


class GSTDCNet(nn.Layer):

    def __init__(self,
                 base=64,
                 layers=[2, 2, 2],
                 block_num=4,
                 type="cat",
                 num_classes=1000,
                 dropout=0.20,
                 use_conv_last=False,
                 pretrained=None):
        super(GSTDCNet, self).__init__()
        if type == "cat":
            block1 = CatBottleneck
            block2 = CatBottleneck_d
        elif type == "add":
            block = AddBottleneck
        self.use_conv_last = use_conv_last
        self.feat_channels = [base // 2, base, base * 4, base * 8, base * 16]
        self.features = self._make_layers(base, layers, block_num, block1)
        self.features_d = self._make_layers(base, layers, block_num, block2)
        self.conv_last = ConvBNRelu(base * 16, max(1024, base * 16), 1, 1)

        if (layers == [4, 5, 3]):  # stdc1446
            self.x2 = nn.Sequential(self.features[:1])  # 0
            self.x4 = nn.Sequential(self.features[1:2])  # 1
            self.x8 = nn.Sequential(self.features[2:6])  # 2-3-4-5
            self.x16 = nn.Sequential(self.features_d[6:11])  # 6-7-8-9-10
            self.x32 = nn.Sequential(self.features_d[11:])  # 11
        elif (layers == [2, 2, 2]):  # stdc813
            self.x2 = nn.Sequential(self.features[:1])  # 0
            self.x4 = nn.Sequential(self.features[1:2])  # 1
            self.x8 = nn.Sequential(self.features[2:4])  # 2 -3
            self.x16 = nn.Sequential(self.features_d[4:6]) # 4-5
            self.x32 = nn.Sequential(self.features_d[6:])  # 6
        else:
            raise NotImplementedError(
                "model with layers:{} is not implemented!".format(layers))

        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        """
        forward function for feature extract.
        """
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        feat8 = self.x8(feat4)
        feat16 = self.x16(feat8)
        feat32 = self.x32(feat16)
        if self.use_conv_last:
            feat32 = self.conv_last(feat32)
        return feat2, feat4, feat8, feat16, feat32

    def _make_layers(self, base, layers, block_num, block):
        features = []
        features += [ConvBNRelu(3, base // 2, 3, 2)]
        features += [ConvBNRelu(base // 2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base * 4, block_num, 2))
                elif j == 0:
                    features.append(
                        block(base * int(math.pow(2, i + 1)), base * int(
                            math.pow(2, i + 2)), block_num, 2))
                else:
                    features.append(
                        block(base * int(math.pow(2, i + 2)), base * int(
                            math.pow(2, i + 2)), block_num, 1))

        return nn.Sequential(*features)

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                param_init.normal_init(layer.weight, std=0.001)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)
        if self.pretrained is not None:
            utils.load_pretrained_model(self, self.pretrained)


class ConvBNRelu(nn.Layer):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, groups=1, dilation=1):
        super(ConvBNRelu, self).__init__()
        self.conv = nn.Conv2D(
            in_planes,
            out_planes,
            kernel_size=kernel,
            stride=stride,
            padding='same',
            groups=groups,
            dilation=dilation,
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
                nn.BatchNorm2D(out_planes // 2), )
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
                nn.BatchNorm2D(out_planes), )
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(
                        in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(
                        out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(
                        out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)), out_planes
                               // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)), out_planes
                               // int(math.pow(2, idx))))

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

class CatBottleneck_d(nn.Layer):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(CatBottleneck_d, self).__init__()
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
                nn.BatchNorm2D(out_planes // 2), )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            self.covn1x1 = nn.Sequential(
            layers.DepthwiseConvBN(in_channels=in_planes, out_channels=in_planes, kernel_size=3, stride=2),
            layers.ConvBN(in_channels=in_planes, out_channels=out_planes, kernel_size=1))
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(
                        in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(
                        out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 4, stride=stride, groups=4, dilation=1))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx + 1)), groups=2, dilation=2))
            else:
                self.conv_list.append(ConvBNRelu(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx)), groups=1, dilation=3))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        # x0=x
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
            # x0 = self.covn1x1(x)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out
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
                nn.BatchNorm2D(out_planes // 2), )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            self.covn1x1 = nn.Sequential(
                layers.DepthwiseConvBN(in_channels=in_planes, out_channels=in_planes, kernel_size=3, stride=2),
                layers.ConvBN(in_channels=in_planes, out_channels=out_planes, kernel_size=1))
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(
                        in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(
                        out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 4, stride=stride, groups=4))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)), out_planes
                               // int(math.pow(2, idx + 1)), groups=2))
            else:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)), out_planes
                               // int(math.pow(2, idx)), groups=1))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        # x0 = x
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
            # x0 = self.covn1x1(x)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out

class ConvBNPRelu(nn.Layer):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, groups=1, dilation=1):
        super(ConvBNPRelu, self).__init__()
        self.conv = nn.Conv2D(
            in_planes,
            out_planes,
            kernel_size=kernel,
            stride=stride,
            padding='same',
            groups=groups,
            dilation=dilation,
            bias_attr=False)
        self.bn = SyncBatchNorm(out_planes, data_format='NCHW')
        self.Prelu = nn.PReLU()

    def forward(self, x):
        out = self.Prelu(self.bn(self.conv(x)))
        return out

@manager.BACKBONES.add_component
def GSTDC2(**kwargs):
    model = GSTDCNet(base=64, layers=[4, 5, 3], **kwargs)
    return model


@manager.BACKBONES.add_component
def GSTDC1(**kwargs):
    model = GSTDCNet(base=64, layers=[2, 2, 2], **kwargs)
    return model
