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
import os
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.cvlibs import manager, param_init
from paddleseg.models import layers
from paddleseg.utils import utils
# import sys
# sys.path.append('/home/aistudio')
# from paddleseg.models import backbones

__all__ = ['sh_lraspp']

@manager.MODELS.add_component
class sh_lraspp(nn.Layer):
    def __init__(self,
                 num_classes,
                 backbone=None,
                 backbone_indices=(2, 4),
                 enable_auxiliary_loss=True,
                 align_corners=False,
                 pretrained=None):
        super().__init__()

        self.backbone = backbone
        # print(type(self.backbone), '321')
        backbone_channels = [
            self.backbone.feat_channels[i] for i in backbone_indices
        ]

        self.head = LR_ASPPHead(num_classes, backbone_indices, backbone_channels, align_corners)
        self.align_corners = align_corners
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        feat_list = self.backbone(x)
        logit_list = self.head(feat_list)
        # print(logit_list[0].shape, logit_list[1].shape)
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


class LR_ASPPHead(nn.Layer):
    def __init__(self, num_classes, backbone_indices, backbone_channels, align_corners):
        super().__init__()
        self.align_corners = align_corners
        self.backbone_indices = backbone_indices
        mid_channels = backbone_channels[0]
        end_channels = backbone_channels[1]
        self.mid_channels =mid_channels
        self.end_channels =end_channels
        self.cbr = layers.ConvBNPReLU(end_channels, 128, kernel_size=1, stride=1)
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(end_channels, 128, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        self.mid_conv_1x1 = nn.Conv2D(mid_channels, num_classes, kernel_size=1, stride=1)
        self.end_conv_1x1 = nn.Conv2D(128, num_classes, kernel_size=1, stride=1)


    def forward(self, feat_list):
 
        low_level_feat = feat_list[0]
        high_level_feat = feat_list[1]
        h_feat = self.cbr(high_level_feat)
        h_feat2 = self.scale(high_level_feat)
        h_feat2 = F.interpolate(h_feat2, paddle.shape(h_feat)[2:], mode="bilinear", align_corners=self.align_corners)
        h_feat = h_feat * h_feat2
        h_feat = F.interpolate(h_feat, paddle.shape(low_level_feat)[2:], mode="bilinear", align_corners=self.align_corners)
        # print("high.shape:", paddle.shape(h_feat))
        # print("low.shape:", paddle.shape(low_level_feat))
        x = paddle.add(self.mid_conv_1x1(low_level_feat), self.end_conv_1x1(h_feat))
        # print("x.shape:", paddle.shape(x))
        return [x]



