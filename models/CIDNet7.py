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
import math
from paddleseg.models.layers.layer_libs import SyncBatchNorm
import os
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.models.backbones.transformer_utils import (DropPath, Identity)
from paddleseg.cvlibs.param_init import (constant_init, kaiming_normal_init,
                                         trunc_normal_init)
from paddle import ParamAttr, reshape, transpose, concat, split
from paddle.nn.initializer import TruncatedNormal, Constant
# from config import *
from paddleseg import utils
from paddleseg.cvlibs import manager, param_init
from paddleseg.models import layers
from functools import partial

BatchNorm2d = paddle.nn.BatchNorm2D
bn_mom = 0.1
# 更改blok, 有残差链接，relu， 更改stem # 增加通道注意力CAM
@manager.MODELS.add_component
class CIDNet7(nn.Layer):
    def __init__(self,
                 # in_channels,
                 num_classes,
                 align_corners=False,
                 pretrained=None):
        super().__init__()

        # mid_channels = 128
        self.mdr = DSBranch()
        self.aux_head1 = SegHead(128, 64, 1)
        self.aux_head2 = SegHead(128, 128, num_classes)
        self.aux_head3 = SegHead(256, 128, num_classes)
        # self.aux_head4 = SegHead(128, 128, num_classes)  # 128
        self.head = SegHead(128, 128, num_classes)  #
        self.align_corners = align_corners
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        pre, x8, x8_, x32 = self.mdr(x)
        # pre = self.mdr(x)

        logit = self.head(pre)

        if not self.training:
            logit_list = [logit]
        else:
            logit1 = self.aux_head1(x8)
            logit2 = self.aux_head2(x8_)
            logit3 = self.aux_head3(x32)
            # logit4 = self.aux_head4(feat4)
            logit_list = [logit, logit1, logit2, logit3]

        logit_list = [
            F.interpolate(
                logit,
                paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=self.align_corners) for logit in logit_list
        ]

        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)
        else:
            for sublayer in self.sublayers():
                if isinstance(sublayer, nn.Conv2D):
                    param_init.kaiming_normal_init(sublayer.weight)
                elif isinstance(sublayer, (nn.BatchNorm, nn.SyncBatchNorm)):
                    param_init.constant_init(sublayer.weight, value=1.0)
                    param_init.constant_init(sublayer.bias, value=0.0)


class ConvBNRelu(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, groups=1):
        super(ConvBNRelu, self).__init__()
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

# 融合，in-   out-
class DSBranch(nn.Layer):

    def __init__(self, pretrained=None):
        super().__init__()

        self.h4 = nn.Sequential(
            ConvBNRelu(128, 128, 3, stride=1, groups=1),
            ConvBNRelu(128, 128, 3, stride=1, groups=1),
        )
        self.h5 = nn.Sequential(
            ConvBNRelu(128, 128, 3, stride=1, groups=1),
            ConvBNRelu(128, 128, 3, stride=1, groups=1),
        )   # 1/8
        self.l1_l2 = nn.Sequential(
                                ConvBNRelu(3, 32, 3, stride=2, groups=1),
                                ConvBNRelu(32, 32, 3, stride=1, groups=1),
                                ConvBNRelu(32, 64, 3, stride=1, groups=1),
                                nn.MaxPool2D(kernel_size=3, stride=2, padding=1),
                                   )
        self.l3 = nn.Sequential(
                                Cat_a(64, 64),
                                CatBottleneck0(in_channels=64, out_channels=128),
                                CatBottleneck0_(in_channels=128, out_channels=128, stride=1),
        )
        self.l4 = nn.Sequential(CatBottleneck0(in_channels=128, out_channels=256),
                                CatBottleneck0_(in_channels=256, out_channels=256, stride=1),
                                )
        self.l5 = nn.Sequential(CatBottleneck0(in_channels=256, out_channels=512),
                                CatBottleneck0_(in_channels=512, out_channels=512, stride=1),
                                )
        self.l6 = nn.Sequential(ConvBNRelu(512, 512, kernel=1),
                                ConvBNRelu(512, 512, kernel=3, stride=1, groups=4),
                                ConvBNRelu(512, 1024, kernel=1))
    # l6是瓶颈块，输出1024
        # 通道变化
        self.compression4 = nn.Sequential(
            nn.Conv2D(256, 128, kernel_size=1, stride=1, bias_attr=False),   # 压缩：256—>128
            SyncBatchNorm(128, data_format='NCHW'),
        )
        self.compression5 = nn.Sequential(
            nn.Conv2D(512, 128, kernel_size=1, stride=1, bias_attr=False),
            SyncBatchNorm(128, data_format='NCHW'),
        )
        #  128—>256
        self.down4 = nn.Sequential(
            nn.Conv2D(128, 256, kernel_size=3, stride=2, padding=1, bias_attr=False),
            SyncBatchNorm(256, data_format='NCHW'),
        )
        self.down5 = nn.Sequential(
            nn.Conv2D(128, 256, kernel_size=3, stride=2, padding=1, bias_attr=False),
            SyncBatchNorm(256, data_format='NCHW'),
            nn.ReLU(),
            nn.Conv2D(256, 512, kernel_size=3, stride=2, padding=1, bias_attr=False),
            SyncBatchNorm(512, data_format='NCHW'),
        )

        self.conv_head32 = ConvBNRelu(256, 128, 3)
        self.conv_smooth256 = ConvBNRelu(256, 256, 3)
        self.conv_smooth128 = ConvBNRelu(128, 128, 3)

        self.relu = nn.ReLU()
        self.spp = SDAPPM(1024, 128, 256)    # 将128改为112？
        self.arm = ARModule(384, 128)
        self.arm1 = ARModule1(128, 128)
        self.pag1 = AlignedModule(128, 64)  # PagFM
        self.pag2 = AlignedModule(128, 64)
        # self.pag2 = PagFM(128, 64)

        # EVCBlock
        # self.evc = EVCBlock(512, 512)
        # self.mlp = Mlp(512, 512)
    # pagFM: DSIM
    # arm5 ：16x 进行一次融合，最后与32x 进行arm融合。
    def forward(self, x):
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8
        # 输入：512×1024
        l2 = self.l1_l2(x)  # c 64,  s 128×256
        l3 = self.l3(l2)    # c 64-->128,  s 64×128
        l4 = self.l4(l3)    # c 128-->256,  s 32×64
        h4 = self.h4(l3)    # c 128-->128,  s 64×128

        l4_ = l4 + self.down4(h4)     # c 128-->256,  s 32×64
        h4_ = self.pag1(h4, self.compression4(l4))              # self.compression4 ：c-128    h4 ：c-128
        h5 = self.h5(self.relu(h4_))    # c ?-->128,  s 64×128
        l5 = self.l5(self.relu(l4_))    # c 256-->512,  s 16×32

        l5_ = l5 + self.down5(h5)       # c 512,  s 16×32
        h5_ = self.pag2(h5, self.compression5(l5))    # c 128

        l6 = self.l6(self.relu(l5_))    # c 512-->1024, s 16×32
        # l6 = self.evc(self.relu(l5_))
        # l6 = self.mlp(self.relu(l5_))
        l7 = self.spp(l6)
        #fusion
        # l4 = self.conv_head16(l4)
        # 获得注意力掩码α
        atten = self.arm(h5_, l7)

        l7_ = self.conv_head32(l7)
        feat_32 = paddle.multiply(l7_, atten) + l7_
        # multiply 相乘，然后再相加
        feat_32_up = F.interpolate(feat_32, size=[height_output, width_output], mode='bilinear')

        feat_8 = paddle.multiply(h5_, (1-atten)) + h5_      # h5_是Fd，辅助分支出来的

        # # 新加嵌套分支##############
        # atten2 = self.sigmoid(out_8_)
        out_8 = feat_32_up + feat_8  # feat_32_up和feat_8都是[2, 128, 64, 128]
        # print(out_8.shape)
        atten2 = self.arm1(out_8)
        # L7_ 需要进行上采样操作，上采样到和 out_8_ 大小一样
        l7_ = F.interpolate(l7_, size=out_8.shape[2:], mode='bilinear', align_corners=True)
        feat_32_ = paddle.multiply(l7_, atten2) + l7_
        feat_8_ = paddle.multiply(h5_, (1 - atten2)) + h5_
        out_8 = feat_32_ + feat_8_
        # ##############################
        out_8 = self.conv_smooth128(out_8)
    # smooth：相当于整合
    #     return out_8

        return out_8, h4_, h5_, l7
    # out_8主损失, h4_, h5_, l7辅助损失


class AlignedModule(nn.Layer):
    def __init__(self, inplane, outplane, kernel_size=3):
        super(AlignedModule, self).__init__()
        self.down_h = nn.Conv2D(inplane, outplane, 1, bias_attr=False)
        self.down_l = nn.Conv2D(inplane, outplane, 1, bias_attr=False)
        self.flow_make = nn.Conv2D(
            outplane * 2,
            2,
            kernel_size=kernel_size,
            padding=1,
            bias_attr=False)
        self.sigmoid = nn.Sigmoid()
        self.upconv = nn.Conv2D(2, outplane, 1, bias_attr=False)
        self.upconv1 = nn.Conv2D(64, 128, 1, bias_attr=False)

    def flow_warp(self, input, flow, size):
        input_shape = paddle.shape(input)  # n, c, h, w = input.shape 对应低分辨率的high-level feature的4个输入维度
        norm = size[::-1].reshape([1, 1, 1, -1])
        norm.stop_gradient = True
        # view(-1, 1) -1表示不确定几行，1表示一列,总长度不变;repeat:维度重复
        # paddle.tile(x, repeat_times) 根据参数 repeat_times 对输入 x 的各维度进行复制
        # 平铺后，输出的第 i 个维度的值等于 x.shape[i]*repeat_times[i]
        h_grid = paddle.linspace(-1.0, 1.0, size[0]).reshape([-1, 1])
        h_grid = h_grid.tile([size[1]])
        w_grid = paddle.linspace(-1.0, 1.0, size[1]).reshape([-1, 1])
        w_grid = w_grid.tile([size[0]]).transpose([1, 0])

        grid = paddle.concat([w_grid.unsqueeze(2), h_grid.unsqueeze(2)], axis=2)
        grid.unsqueeze(0).tile([input_shape[0], 1, 1, 1])
        grid = grid + paddle.transpose(flow, (0, 2, 3, 1)) / norm

        output = F.grid_sample(input, grid)
        return output

        # 注意更换了low_feature, h_feature的顺序

    def forward(self, low_feature, h_feature):
        # low_feature, h_feature = x   low_feature 对应分辨率较高的特征图，h_feature即为低分辨率的high-level feature
        h_feature_orign = h_feature
        size = paddle.shape(low_feature)[2:]
        # 将high-level 和 low-level feature分别通过两个1x1卷积进行压缩
        low_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        # 将high-level feature进行双线性上采样  [2, 64, 64, 128]
        h_feature = F.interpolate(
            h_feature, size=size, mode='bilinear', align_corners=True)
        # 预测语义流场 === 其实就是输入一个3x3的卷积
        flow = self.flow_make(paddle.concat([h_feature, low_feature], 1))

        # 添加相对注意力机制############################
        att = flow
        att = self.upconv(att)
        att = self.sigmoid(att)
        _att = 1 - att  # [2,64,64,128]
        low_feature = low_feature * _att + low_feature  # [2,64,64,128]
        low_feature = self.upconv1(low_feature)  # [2, 128, 64, 128]
        ##############################################
        # 将Flow Field warp 到当前的 high-level feature中
        h_feature_ = self.flow_warp(h_feature_orign, flow, size=size)  # [2, 128, 64, 128]
        h_feature = self.upconv1(h_feature)
        h_feature = h_feature + h_feature_

        h_feature = h_feature + low_feature
        return h_feature


# DSIM 结构
class PagFM(nn.Layer):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False, BatchNorm = BatchNorm2d):
        super(PagFM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
            nn.Conv2D(in_channels, mid_channels,
                      kernel_size=1, bias_attr=False),
            BatchNorm(mid_channels)
        )

        self.f_y = nn.Sequential(
            nn.Conv2D(in_channels, mid_channels,
                      kernel_size=1, bias_attr=False),
            BatchNorm(mid_channels)
        )
        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2D(mid_channels, in_channels,
                          kernel_size=1, bias_attr=False),
                BatchNorm(in_channels)
            )
        if after_relu:
            self.relu = nn.ReLU()
        self.sigmoid_atten = nn.Sigmoid()

    # x是高分辨率分支，y是低分辨率分支
    def forward(self, x, y):
        input_size = paddle.shape(x)
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)

        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]],  # 上采样的目的
                            mode='bilinear', align_corners=False)
        x_k = self.f_x(x)

        if self.with_channel:
            sim_map = self.sigmoid_atten(self.up(x_k * y_q))
        else:
            sim_map = self.sigmoid_atten(paddle.unsqueeze(paddle.sum(x_k * y_q, axis=1), axis=1))
            # unsqueeze:向输入Tensor的Shape中一个或多个位置（axis）插入尺寸为1的维度。
            # 对指定维度上的 Tensor 元素进行求和运算,维度变为1
        y = F.interpolate(y, size=[input_size[2], input_size[3]],
                          mode='bilinear', align_corners=False)

        x = (1 - sim_map) * x + sim_map * y

        return x


# 注意融合模块
class ARModule(nn.Layer):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(ARModule, self).__init__()
        self.conv = ConvBNRelu(in_chan, out_chan, kernel=1, stride=1)  # 输入特征经过k=1，s=1的卷积
        self.conv_atten = nn.Conv2D(out_chan, out_chan, kernel_size=1, bias_attr=False)
        self.bn_atten = BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()
        self.conv1 = nn.Conv2D(128, 256, kernel_size=1,bias_attr=False)
        self.conv2 = nn.Conv2D(256, 384, kernel_size=1, bias_attr=False)

# Fd：h4  Fs:l5 上采样
    def forward(self, h4, l5):

        l5_up = F.interpolate(l5, paddle.shape(h4)[2:], mode='bilinear')  # nearest,bilinear：双线性插值
        # print(l5_up.shape)
        # print(h4.shape)
        h4 = self.conv1(h4)
        # print(h4.shape)
        # fcat = paddle.concat([h4, l5_up], axis=1)  #in-128,256 out-384

        fcat = h4 + l5_up     # in-256,256  out-256
        # print(fcat.shape)
        fcat_ = self.conv2(fcat)
        feat = self.conv(fcat_)

        atten = F.adaptive_avg_pool2d(feat, 1)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        return atten


# 注意融合模块
class ARModule1(nn.Layer):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(ARModule1, self).__init__()
        self.conv = ConvBNRelu(in_chan, out_chan, kernel=1, stride=1)  # 输入特征经过k=1，s=1的卷积
        self.conv_atten = nn.Conv2D(out_chan, out_chan, kernel_size=1, bias_attr=False)
        self.bn_atten = BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

# Fd：h4  Fs:l5 上采样
    def forward(self, l5_):

        # l5_up = F.interpolate(l5, paddle.shape(h4)[2:], mode='bilinear')  # nearest,bilinear：双线性插值
        # fcat = paddle.concat([h4, l5_up], axis=1)
        feat = self.conv(l5_)

        atten = F.adaptive_avg_pool2d(feat, 1)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        return atten


class SegHead(nn.Layer):
    def __init__(self, in_chan, mid_chan, num_classes):
        super(SegHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            layers.ConvBNReLU(in_chan, mid_chan, kernel_size=3))
        self.conv_out = nn.Conv2D(
            mid_chan, num_classes, kernel_size=1, bias_attr=None)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x


class CatConv(nn.Layer):
    def __init__(self, in_channels=128, out_channels=256, stride=2, groups=[8, 4, 2, 1]):
        super(CatConv, self).__init__()

        self.conv1 = ConvBNRelu(in_channels=in_channels, out_channels=out_channels, kernel=3, stride=stride, groups=groups[0])
        self.conv2 = ConvBNRelu(in_channels=out_channels, out_channels=out_channels//2, kernel=3, groups=groups[1])
        self.conv3 = ConvBNRelu(in_channels=out_channels//2, out_channels=out_channels//4, kernel=3, groups=groups[2])
        self.conv4 = ConvBNRelu(in_channels=out_channels//4, out_channels=out_channels//8, kernel=3, groups=groups[3])
        self.conv5 = Conv_BN(in_channels=(out_channels*15)//8, out_channels=out_channels, kernel=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(paddle.concat([x1, x2, x3, x4], axis=1))
        return x5


# Cat_a 没有下采样
class Cat_a(nn.Layer):
    def __init__(self, in_channels=64, out_channels=64, stride=1, groups=[4, 2, 1]):
        super(Cat_a, self).__init__()

        self.conv1 = ConvBNRelu(in_channels=in_channels, out_channels=out_channels//2, kernel=3, groups=groups[0])
        self.conv2 = ConvBNRelu(in_channels=out_channels//2, out_channels=out_channels//4, kernel=3, groups=groups[1])
        self.conv3 = ConvBNRelu(in_channels=out_channels//4, out_channels=out_channels//4, kernel=3, groups=groups[2])
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        out = paddle.concat([x1, x2, x3], axis=1) + x
        return self.relu(out)


class CatBottleneck0_(nn.Layer):
    def __init__(self, in_channels=128, out_channels=128, stride=1, groups=[8, 4, 2, 1]):
        super(CatBottleneck0_, self).__init__()

        self.conv1 = ConvBNRelu(in_channels=in_channels, out_channels=out_channels//2, kernel=1, stride=stride, groups=1)
        self.conv2 = ConvBNRelu(in_channels=out_channels//2, out_channels=out_channels//4, kernel=3, groups=groups[1])
        self.conv3 = ConvBNRelu(in_channels=out_channels//4, out_channels=out_channels//8, kernel=3, groups=groups[2])
        self.conv4 = ConvBNRelu(in_channels=out_channels//8, out_channels=out_channels//8, kernel=3, groups=groups[3])
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        out = paddle.concat([x1, x2, x3, x4], axis=1) + x

        return self.relu(out)


class CatBottleneck0(nn.Layer):
    def __init__(self, in_channels=64, out_channels=256, stride=2, groups=[8, 4, 2, 1]):
        super(CatBottleneck0, self).__init__()

        self.conv1 = ConvBNRelu(in_channels=in_channels, out_channels=out_channels//2, kernel=1, stride=1, groups=1)
        self.conv1_ = Conv_BN(in_channels=out_channels//2, out_channels=out_channels//2, kernel=3, stride=stride, groups=out_channels//2)
        self.conv2 = ConvBNRelu(in_channels=out_channels//2, out_channels=out_channels//4, kernel=3, stride=1, groups=groups[1])
        self.conv3 = ConvBNRelu(in_channels=out_channels//4, out_channels=out_channels//8, kernel=3, stride=1, groups=groups[2])
        self.conv4 = ConvBNRelu(in_channels=out_channels//8, out_channels=out_channels//8, kernel=3, stride=1, groups=groups[3])
        self.avgpool = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
        self.conv1x1 = nn.Sequential(
            layers.DepthwiseConvBN(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2),
            layers.ConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=1))
        self.relu = nn.ReLU()

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.avgpool(x0)
        x0 = self.conv1_(x0)
        x2 = self.conv2(x0)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        out = paddle.concat([x1, x2, x3, x4], axis=1) + self.conv1x1(x)
        return self.relu(out)

class Catbneck_l(nn.Layer):
    def __init__(self, in_channels=128, out_channels=128, stride=1, groups=[4, 2, 1]):
        super(Catbneck_l, self).__init__()

        self.conv1 = ConvBNRelu(in_channels=in_channels, out_channels=out_channels//2, kernel=3, stride=stride, groups=groups[0])
        self.conv2 = ConvBNRelu(in_channels=out_channels//2, out_channels=out_channels//4, kernel=3, groups=groups[1])
        self.conv3 = ConvBNRelu(in_channels=out_channels//4, out_channels=out_channels//8, kernel=3, groups=groups[2])
        self.conv4 = ConvBNRelu(in_channels=out_channels//8, out_channels=out_channels//8, kernel=3, groups=groups[2])
        self.conv5 = ConvBNRelu(in_channels=out_channels*2, out_channels=out_channels, kernel=1)
        self.avgpool = nn.Sequential(nn.AvgPool2D(kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        out = self.conv5(paddle.concat([x, x1, x2, x3, x4], axis=1))
        return out


class Catbneck_h(nn.Layer):
    def __init__(self, in_channels=128, out_channels=128, stride=1, groups=[4, 2, 1]):
        super(Catbneck_h, self).__init__()

        self.conv1 = ConvBNRelu(in_channels=in_channels, out_channels=out_channels//2, kernel=3, stride=stride, groups=groups[0])
        self.conv2 = ConvBNRelu(in_channels=out_channels//2, out_channels=out_channels//4, kernel=3, groups=groups[1])
        self.conv3 = ConvBNRelu(in_channels=out_channels//4, out_channels=out_channels//8, kernel=3, groups=groups[2])
        self.conv4 = ConvBNRelu(in_channels=out_channels//8, out_channels=out_channels//8, kernel=3, groups=groups[2])
        # self.conv5 = Conv_BN(in_channels=out_channels*2, out_channels=out_channels, kernel=1)
        self.conv1x1 = layers.SeparableConvBN(in_channels=in_channels*2, out_channels=out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        out = paddle.concat([x1, x2, x3, x4], axis=1) + x
        out = self.relu(self.conv1x1(paddle.concat([x, out], axis=1)))
        return out
# ======================================================================================================================================================================#
# ======================================================================================================================================================================#

class SDAPPM(nn.Layer):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(SDAPPM, self).__init__()
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

        self.process = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(branch_planes, branch_planes, kernel_size=3, padding=1, bias_attr=False),
        )
        self.compression = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(branch_planes, outplanes, kernel_size=1, bias_attr=False),
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
        x_list.append((F.interpolate(self.scale1(x),
                                                   size=[height, width],
                                                   mode='bilinear')))
        x_list.append(((F.interpolate(self.scale2(x),
                                                    size=[height, width],
                                                    mode='bilinear'))))
        x_list.append((F.interpolate(self.scale3(x),
                                                   size=[height, width],
                                                   mode='bilinear')))
        x_list.append((F.interpolate(self.scale4(x),
                                                   size=[height, width],
                                                   mode='bilinear')))

        out = self.compression(self.process(x_list[0]+x_list[1]+x_list[2]+x_list[3]+x_list[4]))+ self.shortcut(x)
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
# ===============================================================================#
if __name__ == '__main__':
    # import sys
    # sys.path.append('/home/aistudio')
    # from paddleseg.models.backbones import backbones

    image_shape = [2, 1024, 2048]
    x_var = paddle.uniform((1, 3, 1024, 2048), dtype='float32', min=-1., max=1.)

    net = G1_arm3_L3(num_classes=19)
    paddle.summary(net, (1, 3, 1024, 2048))

    y_var = net(x_var)