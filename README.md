# Attention-Nested Dual-Branch Network in Real-time Semantic Segmentation
# Overview
This paper we propose an Attention-Nested Dual-Branch Net- work (ANDNet). This network comprises two parts: the detail branch and the semantic branch, which interact through a Semantic Correspondence Module. This module establishes the positional relation- ships between feature mappings from the detail branch and the semantic branch, thereby improving the pixel misalignment issue. Additionally, we construct a Contextual Nested Attention Fusion Mod- ule to integrate the 
image feature information extracted from the semantic and detail branches, further enhancing the model’s segmentation accuracy.
# 
<img width="898" alt="image" src="https://github.com/user-attachments/assets/12c16881-b2ff-4b0a-abf0-b1e87bbcc4ff" />  

# Training
## Prepare:
- Install gpu driver, cuda toolkit and cudnn
- Install Paddle and PaddleSeg 
- Download dataset and link it to PaddleSeg/data ([Cityscapes](https://paddleseg.bj.bcebos.com/dataset/cityscapes.tar), [CamVid](https://paddleseg.bj.bcebos.com/dataset/camvid.tar))
## Training
The config files of ANDNet are under `configs/GSTDC_Cityscapes`.  
Based on the `train.py` script, we set the config file and start training model.
# Performance 
## Cityscapes
|Model|Backbone|Resolution|GFLOPs|Params|mIoU| 
|-|-|-|-|-|-|
|ANDNet-S|No|512×1024|18.5|6.7M|74.8%|
|ANDNet-L|No|512×1024|18.6|7.2M|75.7%|
|ANDNet-S|No|768×1536|40.1|6.7M|76.3%|
|ANDNet-L|No|768×1536|42.0|7.2M|77.2%|
