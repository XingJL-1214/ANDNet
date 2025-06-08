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
Based on the train.py script, we set the config file and start training model.
