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

import argparse
import random

import paddle
import numpy as np

from paddleseg.cvlibs import manager, Config
from paddleseg.utils import get_sys_env, logger, config_check
from paddleseg.core import train


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')
    # params of training
    parser.add_argument(
        "--config", dest="cfg", help="The config file.", default=None, type=str)
    parser.add_argument(
        '--iters',
        dest='iters',
        help='iters for training',
        type=int,
        default=None)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size of one gpu or cpu',
        type=int,
        default=None)
    parser.add_argument(
        '--learning_rate',
        dest='learning_rate',
        help='Learning rate',
        type=float,
        default=None)
    parser.add_argument(
        '--save_interval',
        dest='save_interval',
        help='How many iters to save a model snapshot once during training.',
        type=int,
        default=1000)
    parser.add_argument(
        '--resume_model',
        dest='resume_model',
        help='The path of resume model',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the model snapshot',
        type=str,
        default='./output')
    parser.add_argument(
        '--keep_checkpoint_max',
        dest='keep_checkpoint_max',
        help='Maximum number of checkpoints to save',
        type=int,
        default=5)
    parser.add_argument(
        '--num_workers',
        dest='num_workers',
        help='Num workers for data loader',
        type=int,
        default=0)
    parser.add_argument(
        '--do_eval',
        dest='do_eval',
        help='Eval while training',
        action='store_true')
    parser.add_argument(
        '--log_iters',
        dest='log_iters',
        help='Display logging information at every log_iters',
        default=20,
        type=int)
    parser.add_argument(
        '--use_vdl',
        dest='use_vdl',
        help='Whether to record the data to VisualDL during training',
        action='store_true')
    parser.add_argument(
        '--seed',
        dest='seed',
        help='Set the random seed during training.',
        default=None,
        type=int)
    parser.add_argument(
        "--precision",
        default="fp32",
        type=str,
        choices=["fp32", "fp16"],
        help="Use AMP (Auto mixed precision) if precision='fp16'. If precision='fp32', the training is normal."
    )
    parser.add_argument(
        "--amp_level",
        default="O1",
        type=str,
        choices=["O1", "O2"],
        help="Auto mixed precision level. Accepted values are “O1” and “O2”: O1 represent mixed precision, the input \
                data type of each operator will be casted by white_list and black_list; O2 represent Pure fp16, all operators \
                parameters and input data will be casted to fp16, except operators in black_list, don’t support fp16 kernel \
                and batchnorm. Default is O1(amp)")
    parser.add_argument(
        '--data_format',
        dest='data_format',
        help='Data format that specifies the layout of input. It can be "NCHW" or "NHWC". Default: "NCHW".',
        type=str,
        default='NCHW')
    parser.add_argument(
        '--profiler_options',
        type=str,
        default=None,
        help='The option of train profiler. If profiler_options is not None, the train ' \
            'profiler is enabled. Refer to the paddleseg/utils/train_profiler.py for details.'
    )
    parser.add_argument(
        '--device',
        dest='device',
        help='Device place to be set, which can be GPU, XPU, NPU, CPU',
        default='gpu',
        type=str)

    return parser.parse_args()


def main(args):

    if args.seed is not None:
        paddle.seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    env_info = get_sys_env()
    info = ['{}: {}'.format(k, v) for k, v in env_info.items()]
    info = '\n'.join(['', format('Environment Information', '-^48s')] + info +
                     ['-' * 48])
    logger.info(info)

    if args.device == 'gpu' and env_info[
            'Paddle compiled with cuda'] and env_info['GPUs used']:
        place = 'gpu'
    elif args.device == 'xpu' and paddle.is_compiled_with_xpu():
        place = 'xpu'
    elif args.device == 'npu' and paddle.is_compiled_with_npu():
        place = 'npu'
    else:
        place = 'cpu'

    paddle.set_device(place)
    if not args.cfg:
        raise RuntimeError('No configuration file specified.')

    cfg = Config(
        args.cfg,
        learning_rate=args.learning_rate,
        iters=args.iters,
        batch_size=args.batch_size)

    # Only support for the DeepLabv3+ model
    if args.data_format == 'NHWC':
        if cfg.dic['model']['type'] != 'DeepLabV3P':
            raise ValueError(
                'The "NHWC" data format only support the DeepLabV3P model!')
        cfg.dic['model']['data_format'] = args.data_format
        cfg.dic['model']['backbone']['data_format'] = args.data_format
        loss_len = len(cfg.dic['loss']['types'])
        for i in range(loss_len):
            cfg.dic['loss']['types'][i]['data_format'] = args.data_format

    train_dataset = cfg.train_dataset
    if train_dataset is None:
        raise RuntimeError(
            'The training dataset is not specified in the configuration file.')
    elif len(train_dataset) == 0:
        raise ValueError(
            'The length of train_dataset is 0. Please check if your dataset is valid'
        )
    val_dataset = cfg.val_dataset if args.do_eval else None
    losses = cfg.loss

    msg = '\n---------------Config Information---------------\n'
    msg += str(cfg)
    msg += '------------------------------------------------'
    logger.info(msg)

    config_check(cfg, train_dataset=train_dataset, val_dataset=val_dataset)

    if place == 'gpu':
        # convert bn to sync_bn
        cfg._model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(cfg.model)

    train(
        cfg.model,
        train_dataset,
        val_dataset=val_dataset,
        optimizer=cfg.optimizer,
        save_dir=args.save_dir,
        iters=cfg.iters,
        batch_size=cfg.batch_size,
        losses=losses,
        resume_model=args.resume_model,
        save_interval=args.save_interval,
        log_iters=args.log_iters,
        num_workers=args.num_workers,
        use_vdl=args.use_vdl,
        keep_checkpoint_max=args.keep_checkpoint_max,
        test_config=cfg.test_config,
        precision=args.precision,
        amp_level=args.amp_level,
        profiler_options=args.profiler_options,
        to_static_training=cfg.to_static_training)


if __name__ == '__main__':
    run_model = 'bisenet'
    args = parse_args()

    if run_model == 'CIDNet':
        args.cfg = '/mnt/HDD_2T/XJL/home_2.5/configs/GSTDC_CamVid/G1_CamVid.yml'
        # args.cfg = '/mnt/HDD_2T/XJL/home_2.5/configs/GSTDC_Cityscapes/G6_Cityscapes.yaml'
        args.save_dir = '/mnt/HDD_2T/XJL/home_2.5/output/a0902/CIDNet6-citypre-small'
        # args.save_dir = '/mnt/HDD_2T/XJL/home_2.5/output/testspeed/a0823-train'
        args.resume_model = '/mnt/HDD_2T/XJL/home_2.5/output/a0902/CIDNet6-citypre-small/iter_24000'
        args.batch_size = 22
        args.num_workers = int(args.batch_size * 1.5)
        args.iters = 60000
        args.do_eval = True
        args.save_interval = 6000
        # use_gpu = True
        main(args)
    if run_model == 'bisenet':
        args.cfg = '/mnt/HDD_2T/XJL/home_2.5/configs/bisenet/bisenet_cityscapes_1024x1024_160k.yml'
        # args.cfg = '/mnt/HDD_2T/XJL/home_2.5/configs/GSTDC_Cityscapes/G6_Cityscapes.yaml'
        args.save_dir = '/mnt/HDD_2T/XJL/home_2.5/output/a0902/CIDNet6-citypre-small'
        # args.save_dir = '/mnt/HDD_2T/XJL/home_2.5/output/testspeed/a0823-train'
        # args.resume_model = '/mnt/HDD_2T/XJL/home_2.5/output/a0902/CIDNet6-citypre-small/iter_24000'
        args.batch_size = 2
        args.num_workers = int(args.batch_size * 1.5)
        args.iters = 60000
        args.do_eval = True
        args.save_interval = 6000
        # use_gpu = True
        main(args)
    elif run_model == 'rtformer':
        args.cfg = 'F:/Seg experiment/home_2.5/configs/rtformer_pag/rtformer_pag_cityscapes_C.yml'
        args.save_dir = 'F:/Seg experiment/home_2.5/output/rtformer/rtf_pag/rtf_pagC-B-1-pre-'
        # args.resume_model = '/home/praiselab/YHT/home_2.5/output/a1119/CNext_50-11-7587/'
        args.batch_size = 20
        args.num_workers = int(args.batch_size * 1.0)
        args.iters = 140000
        args.do_eval = True
        args.save_interval = 6000
        main(args)
    elif run_model == 'G1':
        args.cfg = 'F:/Seg experiment/home_2.5/configs/G1_arm/G1_arm_1024x512.yaml'
        args.save_dir = 'F:/Seg experiment/home_2.5/output/b0302/G1_arm_50-5-DDM-'
        # args.resume_model = '/home/praiselab/YHT/home_2.5/output/a1002/G1_arm_pag2_cam_1024x512-add-/iter_98000'
        args.batch_size = 28
        args.num_workers = int(args.batch_size * 1.5)
        args.iters = 120000
        args.save_interval = 6000
        args.do_eval = True
        main(args)
    elif run_model == 'camvid1':
        args.cfg = 'F:/Seg experiment/home_2.5/configs/G1_arm/G2_arm_camvid.yaml'
        args.save_dir = 'F:/Seg experiment/home_2.5/output/a1013/G2_arm_camvid-3-'
        # args.resume_model = '/home/praiselab/YHT/home_2.5/output/a1002/G1_arm_pag2_cam_1024x512-add-/iter_98000'
        args.batch_size = 8
        args.num_workers = int(args.batch_size * 1.5)
        args.iters = 140000
        args.save_interval = 7000
        args.do_eval = True
        main(args)
    elif run_model == 'coco':
        args.cfg = 'F:/Seg experiment/home_2.5/configs/G1_arm/G1V2_coco.yml'
        args.save_dir = 'F:/Seg experiment/home_2.5/output/a1119/G1_coco-2-'
        # args.resume_model = '/home/praiselab/YHT/home_2.5/output/a1006/lraspp_Mobilenetv2-75-1-/iter_90000'
        args.batch_size = 6
        args.num_workers = int(args.batch_size * 1.5)
        args.iters = 300000
        args.save_interval = 10000
        args.do_eval = True
        main(args)

    elif run_model == 'lraspp_Mobilenetv3':
        args.cfg = 'F:/Seg experiment/home_2.5/configs/lfcn/lraspp_Mobilenetv3.yaml'
        args.save_dir = 'F:/Seg experiment/home_2.5/output/a1006/lraspp_Mobilenetv3-75-1-'
        args.batch_size = 8
        args.num_workers = int(args.batch_size*1.5)
        args.save_interval = 6000
        args.do_eval = True
        main(args)

    elif run_model == 'lraspp_shufflenetv2':
        args.cfg = 'F:/Seg experiment/home_2.5/configs/lfcn/lraspp_shufflenetv2.yaml'
        args.save_dir = 'F:/Seg experiment/home_2.5/output/a0809/lraspp_shufflenetv2-0.5-1-'
        # args.resume_model = '/home/praiselab/YHT/home_2.4/output/a0501/GSTDC1_arm5-0.5-1/iter_147000'
        args.batch_size = 16
        args.num_workers = int(args.batch_size * 1.5)
        args.save_interval = 4000
        args.do_eval = True
        main(args)

    elif run_model == 'STDCNet':
        args.cfg = '/mnt/HDD_2T/XJL/home_2.5/configs/stdcseg/stdc2_seg_cityscapes_1024x512_80k.yml'
        args.save_dir = '/mnt/HDD_2T/XJL/home_2.5/output/stdcseg/a0830'
        # args.resume_model = '/home/praiselab/YHT/home_2.5/output/a0809/lraspp_stdcnet2-0.75-1-/iter_44000'
        args.batch_size = 10
        args.num_workers = int(args.batch_size * 1.5)
        args.save_interval = 6000
        args.do_eval = True
        main(args)
    elif run_model == 'GSTDCNet1':
        args.cfg = 'F:/Seg experiment/home_2.5/configs/lfcn/lraspp_Gstdc.yaml'
        args.save_dir = 'F:/Seg experiment/home_2.5/output/a1019/Gstdc1x2-50-1-'
        args.resume_model = 'F:/Seg experiment/home_2.5/output/a1019/Gstdc1x2-50-1-/iter_76000'
        args.batch_size = 16
        args.iters = 80000
        args.num_workers = int(args.batch_size * 1.5)
        args.save_interval = 4000
        args.do_eval = True
        main(args)
        pass
    else:
        print('please set run_model')