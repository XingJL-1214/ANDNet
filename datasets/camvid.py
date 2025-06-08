# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import sys
import copy
import codecs

sys.path.append('/home/aistudio')
from paddleseg.datasets import Dataset
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose


def add_top(line, dataset_root):
    img, label = line.strip().split(' ')
    # label = label.split('.png')[0] + '_P.png'
    # print([os.path.join(dataset_root, img), os.path.join(dataset_root, label)])
    return [os.path.join(dataset_root, img), os.path.join(dataset_root, label)]


def get_list(path, dataset_root):
    with codecs.open(path, 'r', 'utf-8') as flist:
        lines = [add_top(line, dataset_root) for line in flist]

    return copy.deepcopy(lines)


@manager.DATASETS.add_component
class Camvid(Dataset):
    """
    Cityscapes dataset `https://www.cityscapes-dataset.com/`.
    The folder structure is as follow:

        cityscapes
        |
        |--leftImg8bit
        |  |--train
        |  |--val
        |  |--test
        |
        |--gtFine
        |  |--train
        |  |--val
        |  |--test

    Make sure there are **labelTrainIds.png in gtFine directory. If not, please run the conver_cityscapes.py in tools.

    Args:
        transforms (list): Transforms for image.
        dataset_root (str): Cityscapes dataset directory.
        mode (str, optional): Which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
        edge (bool, optional): Whether to compute edge while training. Default: False
    """
    NUM_CLASSES = 11

    def __init__(self, transforms, dataset_root, mode='train', edge=False):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        self.file_list = list()
        mode = mode.lower()
        self.mode = mode
        self.num_classes = self.NUM_CLASSES
        self.ignore_index = 255
        self.edge = edge

        if mode not in ['train', 'val', 'test']:
            raise ValueError(
                "mode should be 'train', 'val' or 'test', but got {}.".format(
                    mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        # img_dir = os.path.join(self.dataset_root, 'images')
        # label_dir = os.path.join(self.dataset_root, 'labels')
        # if self.dataset_root is None or not os.path.isdir(
        #         self.dataset_root) or not os.path.isdir(
        #             img_dir) or not os.path.isdir(label_dir):
        #     raise ValueError(
        #         "The dataset is not Found or the folder structure is nonconfoumance."
        #     )

        if mode == 'train':
            self.file_list = get_list('datasets/camvid/train_list.txt', self.dataset_root)
        elif mode == 'val':
            self.file_list = get_list('datasets/camvid/test_list.txt', self.dataset_root)
        elif mode == 'test':
            self.file_list = get_list('datasets/camvid/test_list.txt', self.dataset_root)
        else:
            pass

        # print(self.file_list,len(self.file_list))

        # label_files = sorted(
        #     glob.glob(
        #         os.path.join(label_dir, mode, '*',
        #                      '*_P.png')))
        # img_files = sorted(
        #     glob.glob(os.path.join(img_dir, mode, '*', '*.png')))

        # self.file_list = [[
        #     img_path, label_path
        # ] for img_path, label_path in zip(img_files, label_files)]
