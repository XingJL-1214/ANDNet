# ANDNet：Attention-Nested  Dual-Branch  Network  in  Real-time Semantic  Segmentation
# Overview
We propose an Attention-Nested Dual-Branch Net- work (ANDNet). This network comprises two parts: the detail branch and the semantic branch, which interact through a Semantic Correspondence Module. This module establishes the positional relation- ships between feature mappings from the detail branch and the semantic branch, thereby improving the pixel misalignment issue. Additionally, we construct a Contextual Nested Attention Fusion Mod- ule to integrate the image feature information extracted from the semantic and detail branches, further enhancing the model’s segmentation accuracy. We conduct experiments on the Cityscapes and CamVid dataset, and the results demonstrate that our ANDNet achieves a satisfactory balance between segmentation accuracy and inference speed.
<img width="679" alt="image" src="https://github.com/user-attachments/assets/206bf08a-1b89-4399-b3e8-de5f8fda918d" />
# Training
Prepare:
Install gpu driver, cuda toolkit and cudnn
Install Paddle and PaddleSeg 
Download dataset and link it to PaddleSeg/data (Cityscapes, CamVid)
PaddleSeg/data
├── cityscapes
│   ├── gtFine
│   ├── infer.list
│   ├── leftImg8bit
│   ├── test.list
│   ├── train.list
│   ├── trainval.list
│   └── val.list
├── camvid
│   ├── annot
│   ├── images
│   ├── README.md
│   ├── test.txt
│   ├── train.txt
│   └── val.txt
