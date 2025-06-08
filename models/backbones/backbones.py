#N种骨干网【mobilenet、shufflenet、STDCNet、Resnet、】
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import sys
sys.path.append('/home/aistudio')
from paddleseg.models.backbones.layer_libs import ConvBNLayer, BottleneckBlock, BasicBlock
from paddleseg.models.backbones.layer_libs import ConvBNRelu, AddBottleneck, CatBottleneck
from paddleseg.models.backbones.layer_libs import conv_bn, InvertedResidual
from paddleseg.models.backbones.layer_libs import ConvBNLayer3, ResidualUnit
from paddleseg.models.backbones.layer_libs import channel_shuffle, ConvBNLayer_shu, InvertedResidual_shu, InvertedResidualDS
# from paddleseg.models.backbones.layer_libs import ConvBNReLU, Conv_BN, CatConv, CatConv_shortcut, CatConv_down, DAPPM, Catbneck, Catbneck_
from paddleseg.models.backbones.layer_libs import *
from paddleseg.utils import utils
from paddleseg.cvlibs import manager, param_init
from paddleseg.models.layers.layer_libs import SyncBatchNorm
from paddleseg.models.layers import *

@manager.BACKBONES.add_component
class ResNet_vd_x1_0(nn.Layer):
    """
    The ResNet_vd implementation based on PaddlePaddle.

    The original article refers to Jingdong
    Tong He, et, al. "Bag of Tricks for Image Classification with Convolutional Neural Networks"
    (https://arxiv.org/pdf/1812.01187.pdf).

    Args:
        layers (int, optional): The layers of ResNet_vd. The supported layers are (18, 34, 50, 101, 152, 200). Default: 50.
        output_stride (int, optional): The stride of output features compared to input images. It is 8 or 16. Default: 8.
        multi_grid (tuple|list, optional): The grid of stage4. Defult: (1, 1, 1).
        pretrained (str, optional): The path of pretrained model.

    """

    def __init__(self,
                 pretrained=None,
                 data_format='NCHW'):
        super(ResNet_vd_x1_0, self).__init__()
        self.feat_channels =  [64, 64, 128, 256, 512]

        self.stage1 = nn.Sequential(
            ConvBNLayer(in_channels=3, out_channels=32, kernel_size=3, stride=2, act='relu', data_format=data_format),
            ConvBNLayer(in_channels=32, out_channels=32, kernel_size=3, stride=1, act='relu', data_format=data_format),
            ConvBNLayer(in_channels=32, out_channels=64, kernel_size=3, stride=1, act='relu', data_format=data_format),
            nn.MaxPool2D(kernel_size=3, stride=2, padding=1, data_format=data_format),
            )
        
        self.stage2 = nn.Sequential(
            BasicBlock(in_channels=64, out_channels=64, stride=1),
            BasicBlock(in_channels=64, out_channels=64, stride=1),
            )

        self.stage3 = nn.Sequential(
            BasicBlock(in_channels=64, out_channels=128, stride=2, shortcut=False),
            BasicBlock(in_channels=128, out_channels=128, stride=1),
            )
        
        self.stage4 = nn.Sequential(
            BasicBlock(in_channels=128, out_channels=256, stride=2, shortcut=False),
            BasicBlock(in_channels=256, out_channels=256, stride=1),
            )

        self.stage5 = nn.Sequential(
            BasicBlock(in_channels=256, out_channels=512, stride=2, shortcut=False),
            BasicBlock(in_channels=512, out_channels=512, stride=1),
            )

        self.pretrained = pretrained
        self.init_weight()

    def forward(self, X):
        X = self.stage1(X)
        X = self.stage2(X)
        X3 = self.stage3(X)

        X = self.stage4(X3)
        X = self.stage5(X)

        return X3, X

    def init_weight(self):
        utils.load_pretrained_model(self, self.pretrained)


@manager.BACKBONES.add_component
class STDCNet1(nn.Layer):
    def __init__(self,
                 base=64,
                 layers=[2, 2, 2],
                 block_num=4,
                 type="cat",
                 num_classes=1000,
                 dropout=0.20,
                 use_conv_last=True,
                 pretrained=None):
        super(STDCNet1, self).__init__()
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck
        self.use_conv_last = use_conv_last
        self.conv_last = ConvBNRelu(base * 16, max(1024, base * 16), 1, 1)

        self.feat_channels =  [32, 64, 256, 512, 1024]

        self.stage1 = nn.Sequential(
            ConvBNRelu(3, 32, kernel=3, stride=2)
            )
        self.stage2 = nn.Sequential(
            ConvBNRelu(32, 64, kernel=3, stride=2)
            )
        self.stage3 = nn.Sequential(
            CatBottleneck(in_planes=64, out_planes=256, block_num=4, stride=2),
            CatBottleneck(in_planes=256, out_planes=256, block_num=4, stride=1),
            )       
        self.stage4 = nn.Sequential(
            CatBottleneck(in_planes=256, out_planes=512, block_num=4, stride=2),
            CatBottleneck(in_planes=512, out_planes=512, block_num=4, stride=1),
            )
        self.stage5 = nn.Sequential(
            CatBottleneck(in_planes=512, out_planes=1024, block_num=4, stride=2),
            CatBottleneck(in_planes=1024, out_planes=1024, block_num=4, stride=1),
            )

        self.pretrained = pretrained
        self.init_weight()

    def forward(self, X):
        X = self.stage1(X)
        X = self.stage2(X)
        X3 = self.stage3(X)

        X = self.stage4(X3)
        X = self.stage5(X)
        if self.use_conv_last:
            X = self.conv_last(X)
        return X3, X 

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                param_init.normal_init(layer.weight, std=0.001)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)
        if self.pretrained is not None:
            utils.load_pretrained_model(self, self.pretrained)
@manager.BACKBONES.add_component
class STDCNet2(nn.Layer):
    def __init__(self,
                 base=64,
                 layers=[2, 2, 2],
                 block_num=4,
                 type="cat",
                 num_classes=1000,
                 dropout=0.20,
                 use_conv_last=True,
                 pretrained=None):
        super(STDCNet2, self).__init__()
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck
        self.use_conv_last = use_conv_last
        self.conv_last = ConvBNRelu(base * 16, max(1024, base * 16), 1, 1)

        self.feat_channels =  [32, 64, 256, 512, 1024]

        self.stage1 = nn.Sequential(
            ConvBNRelu(3, 32, kernel=3, stride=2)
            )
        self.stage2 = nn.Sequential(
            ConvBNRelu(32, 64, kernel=3, stride=2)
            )
        self.stage3 = nn.Sequential(
            CatBottleneck(in_planes=64, out_planes=256, block_num=4, stride=2),
            CatBottleneck(in_planes=256, out_planes=256, block_num=4, stride=1),
            CatBottleneck(in_planes=256, out_planes=256, block_num=4, stride=1),
            CatBottleneck(in_planes=256, out_planes=256, block_num=4, stride=1),
            )
        self.stage4 = nn.Sequential(
            CatBottleneck(in_planes=256, out_planes=512, block_num=4, stride=2),
            CatBottleneck(in_planes=512, out_planes=512, block_num=4, stride=1),
            CatBottleneck(in_planes=512, out_planes=512, block_num=4, stride=1),
            CatBottleneck(in_planes=512, out_planes=512, block_num=4, stride=1),
            CatBottleneck(in_planes=512, out_planes=512, block_num=4, stride=1),
            )
        self.stage5 = nn.Sequential(
            CatBottleneck(in_planes=512, out_planes=1024, block_num=4, stride=2),
            CatBottleneck(in_planes=1024, out_planes=1024, block_num=4, stride=1),
            CatBottleneck(in_planes=1024, out_planes=1024, block_num=4, stride=1),
            )

        self.pretrained = pretrained
        self.init_weight()

    def forward(self, X):
        X = self.stage1(X)
        X = self.stage2(X)
        X3 = self.stage3(X)

        X = self.stage4(X3)
        X = self.stage5(X)
        if self.use_conv_last:
            X = self.conv_last(X)
        return X3, X

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                param_init.normal_init(layer.weight, std=0.001)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)
        if self.pretrained is not None:
            utils.load_pretrained_model(self, self.pretrained)
# M2
def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

@manager.BACKBONES.add_component
class MobileNetV2_x1(nn.Layer):
    def __init__(self, channel_ratio=1.0, min_channel=16, use_conv_last=False, pretrained=None):
        super(MobileNetV2_x1, self).__init__()
        self.channel_ratio = channel_ratio
        self.min_channel = min_channel
        self.pretrained = pretrained
        self.use_conv_last = use_conv_last
        
        self.feat_channels =  [32, 24, 32, 96, 320]
        # self.feat_channels[-1] =  1280 if self.use_conv_last else 320
        # self.conv_last = conv_bn(self.depth(320), self.depth(1280), 1, 1)
        self.stage1 = nn.Sequential(
            conv_bn(3, self.depth(32), 3, 2),
            InvertedResidual(self.depth(32), self.depth(16), 1, 1)
            )      
        self.stage2 = nn.Sequential(
            InvertedResidual(self.depth(16), self.depth(24), 2, 6),
            InvertedResidual(self.depth(24), self.depth(24), 1, 6),
            )
        self.stage3 = nn.Sequential(
            InvertedResidual(self.depth(24), self.depth(32), 2, 6),
            InvertedResidual(self.depth(32), self.depth(32), 1, 6),
            InvertedResidual(self.depth(32), self.depth(32), 1, 6),
            )
        self.stage4 = nn.Sequential(
            InvertedResidual(self.depth(32), self.depth(64), 2, 6),
            InvertedResidual(self.depth(64), self.depth(64), 1, 6),
            InvertedResidual(self.depth(64), self.depth(64), 1, 6),
            InvertedResidual(self.depth(64), self.depth(64), 1, 6),
            InvertedResidual(self.depth(64), self.depth(96), 1, 6),
            InvertedResidual(self.depth(96), self.depth(96), 1, 6),
            InvertedResidual(self.depth(96), self.depth(96), 1, 6),
            )
        self.stage5 = nn.Sequential(
            InvertedResidual(self.depth(96), self.depth(160), 2, 6),
            InvertedResidual(self.depth(160), self.depth(160), 1, 6),
            InvertedResidual(self.depth(160), self.depth(160), 1, 6),
            InvertedResidual(self.depth(160), self.depth(320), 1, 6),
            )
        self.init_weight()

    def depth(self, channels):
        min_channel = min(channels, self.min_channel)
        return max(min_channel, int(channels * self.channel_ratio))

    def forward(self, X):
        X = self.stage1(X)
        X = self.stage2(X)
        X3 = self.stage3(X)

        X = self.stage4(X3)
        X = self.stage5(X)
        if self.use_conv_last:
            X = self.conv_last(X)
        return X3, X 

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

# M3
@manager.BACKBONES.add_component
class MobileNetV3_s1(nn.Layer):

    def __init__(self,
                 pretrained=None,
                 scale=1.0,
                 model_name="small",
                 use_conv_last=True,
                 output_stride=None):
        super(MobileNetV3_s1, self).__init__()
        inplanes = 16
        self.use_conv_last = use_conv_last
        
        self.feat_channels =  [16, 16, 24, 48, 96]
        self.feat_channels[-1] =  576 if self.use_conv_last else 96
        self.conv_last = ConvBNLayer3(in_c=96, out_c=576, filter_size=1, 
            stride=1, padding=1, num_groups=1, if_act=True, act="hard_swish")

        self.stage1 = nn.Sequential(
            ConvBNLayer3(in_c=3, out_c=make_divisible(16), filter_size=3, 
            stride=2, padding=1, num_groups=1, if_act=True, act="hard_swish")
            ) 
        self.stage2 = nn.Sequential(
            ResidualUnit(in_c=16, mid_c=16, out_c=16, filter_size=3, stride=2, use_se=True, act='relu')
            )      
        self.stage3 = nn.Sequential(
            ResidualUnit(in_c=16, mid_c=72, out_c=24, filter_size=3, stride=2, use_se=False, act='relu'),
            ResidualUnit(in_c=24, mid_c=88, out_c=24, filter_size=3, stride=2, use_se=False, act='relu'),
            )
        self.stage4 = nn.Sequential(
            ResidualUnit(in_c=24, mid_c=96, out_c=40, filter_size=5, stride=2, use_se=True, act='hard_swish'),
            ResidualUnit(in_c=40, mid_c=240, out_c=40, filter_size=5, stride=1, use_se=True, act='hard_swish'),
            ResidualUnit(in_c=40, mid_c=240, out_c=40, filter_size=5, stride=1, use_se=True, act='hard_swish'),
            ResidualUnit(in_c=40, mid_c=120, out_c=48, filter_size=5, stride=1, use_se=True, act='hard_swish'),
            ResidualUnit(in_c=48, mid_c=144, out_c=48, filter_size=5, stride=1, use_se=True, act='hard_swish')
            )
        self.stage5 = nn.Sequential(
            ResidualUnit(in_c=48, mid_c=288, out_c=96, filter_size=5, stride=2, use_se=True, act='hard_swish'),
            ResidualUnit(in_c=96, mid_c=576, out_c=96, filter_size=5, stride=1, use_se=True, act='hard_swish'),
            ResidualUnit(in_c=96, mid_c=576, out_c=96, filter_size=5, stride=1, use_se=True, act='hard_swish')
            )
        
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, X):
        X = self.stage1(X)
        X = self.stage2(X)
        X3 = self.stage3(X)

        X = self.stage4(X3)
        X = self.stage5(X)
        if self.use_conv_last:
            X = self.conv_last(X)
        return X3, X 

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_pretrained_model(self, self.pretrained)

# ShuffleNetv2
@manager.BACKBONES.add_component
class ShuffleNetV2_x1(nn.Layer):
    def __init__(self, use_conv_last=False, pretrained=None, scale=1.0, act="relu"):
        super(ShuffleNetV2_x1, self).__init__()
        self.scale = scale
        self.use_conv_last = use_conv_last
        stage_out_channels = []
        stage_repeats = [4, 8, 4]
        if scale == 0.25:
            stage_out_channels = [24, 24, 48, 96, 512]
        elif scale == 0.33:
            stage_out_channels = [24, 24, 32, 64, 128, 512]
        elif scale == 0.5:
            stage_out_channels = [24, 24, 48, 96, 192]
        elif scale == 1.0:
            stage_out_channels = [24, 24, 116, 232, 464]
        elif scale == 1.5:
            stage_out_channels = [24, 24, 176, 352, 704]
        elif scale == 2.0:
            stage_out_channels = [24, 24, 224, 488, 976]
        else:
            raise NotImplementedError("This scale size:[" + str(scale) +
                                      "] is not implemented!")
        
        self.feat_channels =  stage_out_channels
        if self.use_conv_last:
            stage_out_channels.append(1024)

        self.stage1_2 = nn.Sequential(
            ConvBNLayer_shu(in_channels=3, out_channels=stage_out_channels[1], kernel_size=3, 
            stride=2, padding=1, act=act, name='stage1_conv'),
            nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
            )      
        self.stage3 = nn.Sequential(
            InvertedResidualDS(in_channels=stage_out_channels[1], out_channels=stage_out_channels[2], stride=2, act=act, name='2_1'),
            InvertedResidual_shu(in_channels=stage_out_channels[2], out_channels=stage_out_channels[2], stride=1, act=act, name='2_2'),
            InvertedResidual_shu(in_channels=stage_out_channels[2], out_channels=stage_out_channels[2], stride=1, act=act, name='2_3'),
            InvertedResidual_shu(in_channels=stage_out_channels[2], out_channels=stage_out_channels[2], stride=1, act=act, name='2_4'),
            )
        self.stage4 = nn.Sequential(
            InvertedResidualDS(in_channels=stage_out_channels[2], out_channels=stage_out_channels[3], stride=2, act=act, name='3_1'),
            InvertedResidual_shu(in_channels=stage_out_channels[3], out_channels=stage_out_channels[3], stride=1, act=act, name='3_2'),
            InvertedResidual_shu(in_channels=stage_out_channels[3], out_channels=stage_out_channels[3], stride=1, act=act, name='3_3'),
            InvertedResidual_shu(in_channels=stage_out_channels[3], out_channels=stage_out_channels[3], stride=1, act=act, name='3_4'),
            InvertedResidual_shu(in_channels=stage_out_channels[3], out_channels=stage_out_channels[3], stride=1, act=act, name='3_5'),
            InvertedResidual_shu(in_channels=stage_out_channels[3], out_channels=stage_out_channels[3], stride=1, act=act, name='3_6'),
            InvertedResidual_shu(in_channels=stage_out_channels[3], out_channels=stage_out_channels[3], stride=1, act=act, name='3_7'),
            InvertedResidual_shu(in_channels=stage_out_channels[3], out_channels=stage_out_channels[3], stride=1, act=act, name='3_8'),
            )
        self.stage5 = nn.Sequential(
            InvertedResidualDS(in_channels=stage_out_channels[3], out_channels=stage_out_channels[4], stride=2, act=act, name='4_1'),
            InvertedResidual_shu(in_channels=stage_out_channels[4], out_channels=stage_out_channels[4], stride=1, act=act, name='4_2'),
            InvertedResidual_shu(in_channels=stage_out_channels[4], out_channels=stage_out_channels[4], stride=1, act=act, name='4_3'),
            InvertedResidual_shu(in_channels=stage_out_channels[4], out_channels=stage_out_channels[4], stride=1, act=act, name='4_4'),
            )

        # 3. last_conv
        self.conv_last = ConvBNLayer_shu(
            in_channels=stage_out_channels[-2],out_channels=stage_out_channels[-1], kernel_size=1, stride=1, padding=0, act=act, name='conv5')



    def forward(self, X):
        X = self.stage1_2(X)
        X3 = self.stage3(X)

        X = self.stage4(X3)
        X = self.stage5(X)
        if self.use_conv_last:
            X = self.conv_last(X)
        return X3, X 



@manager.MODELS.add_component
class GSTDCNet_lraspp_1X2(nn.Layer):

    def __init__(self, align_corners=False, pretrained=None):
        super(GSTDCNet_lraspp_1X2, self).__init__()
        self.feat_channels =  [64, 64, 256, 512, 1024]
        self.h4 = nn.Sequential(
            ConvBNRelu(128, 128, 3, stride=1, groups=1),
            ConvBNRelu(128, 128, 3, stride=1, groups=1),
        )
        self.h5 = nn.Sequential(
            ConvBNRelu(128, 128, 3, stride=1, groups=1),
            ConvBNRelu(128, 128, 3, stride=1, groups=1),
        )
        self.stage1 = nn.Sequential(
            ConvBNRelu(3, 32, 3, stride=2, groups=1),
            ConvBNRelu(32, 32, 3, stride=1, groups=1),
            ConvBNRelu(32, 64, 3, stride=1, groups=1),
            )
        self.stage2 = nn.Sequential(
            nn.MaxPool2D(kernel_size=3, stride=2, padding=1),
            Cat_a(64, 64),
            )
        self.stage3 = nn.Sequential(
            GCatBottleneck(in_channels=64, out_channels=256),
            GCatBottleneck_(in_channels=256, out_channels=256, stride=1)
            )
        self.stage4 = nn.Sequential(
            GCatBottleneck(in_channels=256, out_channels=512),
            GCatBottleneck_(in_channels=512, out_channels=512, stride=1)
            )
        self.stage5 = nn.Sequential(
            GCatBottleneck(in_channels=512, out_channels=1024),
            GCatBottleneck_(in_channels=1024, out_channels=1024, stride=1)
            )
        self.l6 = nn.Sequential(ConvBNRelu(512, 512, kernel=1),
                                ConvBNRelu(512, 512, kernel=3, stride=1, groups=4),
                                ConvBNRelu(512, 1024, kernel=1))
        self.spp = SDAPPM(1024, 128, 256)

    def forward(self, X):
        X = self.stage1(X)
        X = self.stage2(X)
        X3 = self.stage3(X)
        # X3 = self.h4(X3)
        # X3 = self.h5(X3)

        X = self.stage4(X3)
        X = self.stage5(X)
        # X = self.l6(X)
        # X = self.spp(X)

        return X3, X
@manager.MODELS.add_component
class GSTDCNet_lraspp_2X1(nn.Layer):

    def __init__(self, align_corners=False, pretrained=None):
        super(GSTDCNet_lraspp_2X1, self).__init__()
        self.feat_channels =  [64, 64, 128, 256, 512]

        self.stage1 = nn.Sequential(
            ConvBNRelu(3, 32, 3, stride=2, groups=1),
            ConvBNRelu(32, 32, 3, stride=1, groups=1),
            ConvBNRelu(32, 64, 3, stride=1, groups=1),
        )
        self.stage2 = nn.Sequential(
            nn.MaxPool2D(kernel_size=3, stride=2, padding=1),
            Cat_a(64, 64),
        )
        self.stage3 = nn.Sequential(
            GCatBottleneck(in_channels=64, out_channels=128),
            GCatBottleneck_(in_channels=128, out_channels=128, stride=1),
            GCatBottleneck_(in_channels=128, out_channels=128, stride=1),
            GCatBottleneck_(in_channels=128, out_channels=128, stride=1)
            )
        self.stage4 = nn.Sequential(
            GCatBottleneck(in_channels=128, out_channels=256),
            GCatBottleneck_(in_channels=256, out_channels=256, stride=1),
            GCatBottleneck_(in_channels=256, out_channels=256, stride=1),
            GCatBottleneck_(in_channels=256, out_channels=256, stride=1),
            GCatBottleneck_(in_channels=256, out_channels=256, stride=1),
            )
        self.stage5 = nn.Sequential(
            GCatBottleneck(in_channels=256, out_channels=512),
            GCatBottleneck_(in_channels=512, out_channels=512, stride=1),
            GCatBottleneck_(in_channels=512, out_channels=512, stride=1),

            )

    def forward(self, X):
        X = self.stage1(X)
        X = self.stage2(X)
        X3 = self.stage3(X)

        X = self.stage4(X3)
        X = self.stage5(X)

        return X3, X
@manager.MODELS.add_component
class GSTDCNet_lraspp_1X1(nn.Layer):

    def __init__(self, align_corners=False, pretrained=None):
        super(GSTDCNet_lraspp_1X1, self).__init__()
        self.feat_channels =  [32, 64, 128, 256, 512]

        self.stage1 = nn.Sequential(
            ConvBNRelu(3, 32, 3, stride=2, groups=1),
            # ConvBNRelu(32, 32, 3, stride=1, groups=1),

            )
        self.stage2 = nn.Sequential(

            nn.MaxPool2D(kernel_size=3, stride=2, padding=1),
            ConvBNRelu(32, 64, 3, stride=2, groups=1),
            # ConvBNRelu(64, 64, 3, stride=2, groups=1),
            Cat_a(64, 64),
            )
        self.stage3 = nn.Sequential(
            GCatBottleneck(in_channels=64, out_channels=128),
            GCatBottleneck_(in_channels=128, out_channels=128, stride=1)
            )
        self.stage4 = nn.Sequential(
            GCatBottleneck(in_channels=128, out_channels=256),
            GCatBottleneck_(in_channels=256, out_channels=256, stride=1)
            )
        self.stage5 = nn.Sequential(
            GCatBottleneck(in_channels=256, out_channels=512),
            GCatBottleneck_(in_channels=512, out_channels=512, stride=1)
            )

    def forward(self, X):
        X = self.stage1(X)
        X = self.stage2(X)
        X3 = self.stage3(X)

        X = self.stage4(X3)
        X = self.stage5(X)

        return X3, X
@manager.MODELS.add_component
class GSTDCNet_lraspp_2X2(nn.Layer):

    def __init__(self, align_corners=False, pretrained=None):
        super(GSTDCNet_lraspp_2X2, self).__init__()
        self.feat_channels =  [32, 64, 256, 512, 1024]

        self.stage1 = nn.Sequential(
            ConvBNRelu(3, 32, 3, stride=2, groups=1),
            ConvBNRelu(32, 32, 3, stride=1, groups=1),
            ConvBNRelu(32, 64, 3, stride=1, groups=1),
            )
        self.stage2 = nn.Sequential(
            nn.MaxPool2D(kernel_size=3, stride=2, padding=1),
            # ConvBNRelu(32, 64, 3, stride=2, groups=1),
            Cat_a(64, 64),
            )
        self.stage3 = nn.Sequential(
            GCatBottleneck(in_channels=64, out_channels=256),
            GCatBottleneck_(in_channels=256, out_channels=256, stride=1),
            GCatBottleneck_(in_channels=256, out_channels=256, stride=1),
            GCatBottleneck_(in_channels=256, out_channels=256, stride=1),
            )
        self.stage4 = nn.Sequential(
            GCatBottleneck(in_channels=256, out_channels=512),
            GCatBottleneck_(in_channels=512, out_channels=512, stride=1),
            GCatBottleneck_(in_channels=512, out_channels=512, stride=1),
            GCatBottleneck_(in_channels=512, out_channels=512, stride=1),
            GCatBottleneck_(in_channels=512, out_channels=512, stride=1),

            )
        self.stage5 = nn.Sequential(
            GCatBottleneck(in_channels=512, out_channels=1024),
            GCatBottleneck_(in_channels=1024, out_channels=1024, stride=1),
            GCatBottleneck_(in_channels=1024, out_channels=1024, stride=1)
            )

    def forward(self, X):
        X = self.stage1(X)
        X = self.stage2(X)
        X3 = self.stage3(X)

        X = self.stage4(X3)
        X = self.stage5(X)

        return X3, X

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
class Cat_a(nn.Layer):
    def __init__(self, in_channels=64, out_channels=64, stride=1, groups=[4, 2, 1]):
        super(Cat_a, self).__init__()

        self.conv1 = ConvBNRelu(in_channels=in_channels, out_channels=out_channels//2, kernel=3, groups=1)
        self.conv2 = ConvBNRelu(in_channels=out_channels//2, out_channels=out_channels//4, kernel=3, groups=1)
        self.conv3 = ConvBNRelu(in_channels=out_channels//4, out_channels=out_channels//8, kernel=3, groups=1)
        self.conv4 = ConvBNRelu(in_channels=out_channels // 8, out_channels=out_channels // 8, kernel=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        out = paddle.concat([x1, x2, x3, x4], axis=1)+x
        return self.relu(out)

class Cat(nn.Layer):
    def __init__(self, in_channels=128, out_channels=128, stride=1, groups=[8, 4, 2, 1]):
        super(Cat, self).__init__()

        self.conv1 = ConvBNRelu(in_channels=in_channels, out_channels=out_channels//2, kernel=1, stride=stride, groups=1)
        self.conv2 = ConvBNRelu(in_channels=out_channels//2, out_channels=out_channels//4, kernel=3)
        self.conv3 = ConvBNRelu(in_channels=out_channels//4, out_channels=out_channels//8, kernel=3)
        self.conv4 = ConvBNRelu(in_channels=out_channels//8, out_channels=out_channels//8, kernel=3)
        # self.conv5 = Conv_BN(in_channels=out_channels*2, out_channels=out_channels, kernel=1)
        # self.conv1x1 = layers.SeparableConvBN(in_channels=in_channels*2, out_channels=out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        out = paddle.concat([x1, x2, x3, x4], axis=1) + x
        # out = self.relu(self.conv1x1(paddle.concat([x, out], axis=1)))
        return self.relu(out)
class GCatBottleneck_(nn.Layer):
    def __init__(self, in_channels=128, out_channels=128, stride=1, groups=[8, 4, 2, 1]):
        super(GCatBottleneck_, self).__init__()

        self.conv1 = ConvBNRelu(in_channels=in_channels, out_channels=out_channels//2, kernel=1, stride=stride, groups=1)
        self.conv2 = ConvBNRelu(in_channels=out_channels//2, out_channels=out_channels//4, kernel=3, groups=groups[1])
        self.conv3 = ConvBNRelu(in_channels=out_channels//4, out_channels=out_channels//8, kernel=3, groups=groups[2])
        self.conv4 = ConvBNRelu(in_channels=out_channels//8, out_channels=out_channels//8, kernel=3, groups=groups[3])
        # self.conv5 = Conv_BN(in_channels=out_channels*2, out_channels=out_channels, kernel=1)
        # self.conv1x1 = layers.SeparableConvBN(in_channels=in_channels*2, out_channels=out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        out = paddle.concat([x1, x2, x3, x4], axis=1) + x
        # out = self.relu(self.conv1x1(paddle.concat([x, out], axis=1)))
        return self.relu(out)
class GCatBottleneck(nn.Layer):
    def __init__(self, in_channels=64, out_channels=256, stride=2, groups=[8, 4, 2, 1]):
        super(GCatBottleneck, self).__init__()

        self.conv1 = ConvBNRelu(in_channels=in_channels, out_channels=out_channels//2, kernel=1, stride=1, groups=1)
        self.conv1_ = Conv_BN(in_channels=out_channels//2, out_channels=out_channels//2, kernel=3, stride=stride, groups=out_channels//2)
        self.conv2 = ConvBNRelu(in_channels=out_channels//2, out_channels=out_channels//4, kernel=3, stride=1, groups=groups[1])
        self.conv3 = ConvBNRelu(in_channels=out_channels//4, out_channels=out_channels//8, kernel=3, stride=1, groups=groups[2])
        self.conv4 = ConvBNRelu(in_channels=out_channels//8, out_channels=out_channels//8, kernel=3, stride=1, groups=groups[3])
        # self.conv5 = Conv_BN(in_channels=out_channels*2, out_channels=out_channels, kernel=1)
        self.avgpool = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
        # self.conv1x1 = layers.SeparableConvBNReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.shortcut1 = nn.Sequential(
            layers.DepthwiseConvBN(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2),
            layers.ConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=1))
        self.shortcut2 = nn.Sequential(
            nn.AvgPool2D(kernel_size=3, stride=2, padding=1),
            layers.ConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.avgpool(x0)
        x0 = self.conv1_(x0)
        x2 = self.conv2(x0)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        out = paddle.concat([x1, x2, x3, x4], axis=1) + self.shortcut1(x)
        # out = self.relu(self.conv1x1(paddle.concat([x, out], axis=1)))
        return self.relu(out)


















if __name__ == '__main__':
    # import sys
    # sys.path.append('/home/aistudio')
    # from paddleseg.models.backbones import backbones

    image_shape = [2, 1024, 2048]
    x_var = paddle.uniform((1, 3, 1024, 2048), dtype='float32', min=-1., max=1.)

    net = DualSDRv1_x1()
    paddle.summary(net, (1, 3, 1024, 2048))
    
    y_var = net(x_var)
    
    # print(len(y_var))
    # for y in y_var:
    #     y = y.numpy()
        # print('logit:', y.shape)