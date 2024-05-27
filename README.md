
<div align="center">
<h1>vHeat</h1>
<h3>vHeat: Building Vision Models upon Heat Conduction</h3>

[ZhaoZhi Wang](https://scholar.google.com/citations?user=CkDanj8AAAAJ&hl=zh-CN&oi=ao)<sup>1,2*</sup>, [Yue Liu](https://github.com/MzeroMiko)<sup>1*</sup>, [Yunfan Liu](https://scholar.google.com.hk/citations?user=YPL33G0AAAAJ&hl=zh-CN&oi=ao)<sup>1</sup>, [Hongtian Yu](https://github.com/yuhongtian17)<sup>1</sup>, 

[Yaowei Wang](https://scholar.google.com.hk/citations?user=o_DllmIAAAAJ&hl=zh-CN&oi=ao)<sup>2,3</sup>, [Qixiang Ye](https://scholar.google.com.hk/citations?user=tjEfgsEAAAAJ&hl=zh-CN&oi=ao)<sup>1,2</sup>, [Yunjie Tian](https://sunsmarterjie.github.io/)<sup>1</sup>

<sup>1</sup> University of Chinese Academy of Sciences, <sup>2</sup> Peng Cheng Laboratory,

<sup>3</sup> Harbin Institute of Technology (Shenzhen)

<sup>*</sup> Equal Contributions

Paper: ([?](?))

</div>

## Abstract
A fundamental problem in learning robust and expressive visual representations lies in efficiently estimating the spatial relationships of visual semantics throughout the entire image. In this study, we propose vHeat, a novel vision backbone model that simultaneously achieves both high computational efficiency and global receptive field. The essential idea, inspired by the physical principle of heat conduction, is to conceptualize image patches as heat sources and model the calculation of their correlations as the diffusion of thermal energy. This mechanism is incorporated into deep models through the newly proposed module, the Heat Conduction Operator (HCO), which is physically plausible and can be efficiently implemented using DCT and IDCT operations with a complexity of O(N<sup>1.5</sup>). Extensive experiments demonstrate that vHeat surpasses Vision Transformers (ViTs) across various vision tasks, while also providing higher inference speeds, reduced FLOPs, and lower GPU memory usage for high-resolution images. 

## Main Results

:book: 
***Checkpoint and log files will be released soon***

### **Classification on ImageNet-1K with vHeat**

| name | pretrain | resolution |acc@1 | #params | FLOPs | Throughput | configs/logs/ckpts |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---:|
| Swin-T | ImageNet-1K | 224x224 | 81.2 | 29M | 4.5G | 1244 | 
| Swin-S | ImageNet-1K | 224x224 | 83.0 | 50M | 8.7G | 728 |
| Swin-B | ImageNet-1K | 224x224 | 83.5 | 89M | 15.4G | 458 |
| vHeat-T | ImageNet-1K | 224x224 | 82.2 | 29M | 4.6G | 1514 | [config](classification/configs/vHeat/vHeat_tiny_224.yaml)/[log](#)/[ckpt](#) |
| vHeat-S | ImageNet-1K | 224x224 | 83.6 | 50M | 8.5G | 945 | [config](classification/configs/vHeat/vHeat_small_224.yaml)/[log](#)/[ckpt](#) |
| vHeat-B | ImageNet-1K | 224x224 | 83.9 | 87M | 14.9G | 661 | [config](classification/configs/vHeat/vHeat_base_224.yaml)/[log](#)/[ckpt](#) |

* *Models in this subsection is trained from scratch with random or manual initialization.*
* *Throughput is test on `pytorch2.0 + cuda12.1 + A100 + AMD EPYC 7542 CPU`.*
* *We use ema because our model is still under development.*

### **Object Detection on COCO with vHeat**
  
| Backbone | #params | FLOPs | Detector | box mAP | mask mAP | configs/logs/ckpts |
| :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| Swin-T | 48M | 267G | MaskRCNN@1x | 42.7| 39.3 |-- |-- |
| vHeat-T | 53M | 286G | MaskRCNN@1x | 45.1| 41.0 | [config](detection/configs/vheat/mask_rcnn_vssm_fpn_coco_tiny.py)/[log](#)/[ckpt](#) |
| Swin-S | 69M | 354G | MaskRCNN@1x | 44.8| 40.9 |-- |-- |
| vHeat-S | 74M | 377G | MaskRCNN@1x | 46.8| 42.3 | [config](detection/configs/vheat/mask_rcnn_vssm_fpn_coco_small.py)/[log](#)/[ckpt](#) |
| Swin-B | 107M | 496G | MaskRCNN@1x | 46.9| 42.3 |-- |-- |
| vHeat-B | 115M | 526G | MaskRCNN@1x | 47.7 | 43.0 | [config](detection/configs/vheat/mask_rcnn_vssm_fpn_coco_base.py)/[log](#)/[ckpt](#) |
| Swin-T | 48M | 267G | MaskRCNN@3x | 46.0| 41.6 |-- |-- |
| vHeat-T | 53M | 286G | MaskRCNN@3x | 47.2| 42.4 | [config](detection/configs/vheat/mask_rcnn_vssm_fpn_coco_tiny1_ms_3x.py)/[log](h#)/[ckpt](#) |
| Swin-S | 69M | 354G | MaskRCNN@3x | 48.2| 43.2 |-- |-- |
| vHeat-S | 74M | 377G | MaskRCNN@3x | 48.8| 43.7 | [config](detection/configs/vheat/mask_rcnn_vssm_fpn_coco_small_ms_3x.py)/[log](#)/[ckpt](#) |

* *Models in this subsection is initialized from the models trained in `classfication`.*


### **Semantic Segmentation on ADE20K with vHeat**

| Backbone | Input|  #params | FLOPs | Segmentor | mIoU(SS) | configs/logs/ckpts |
| :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| Swin-T | 512x512 | 60M | 945G | UperNet@160k | 44.4| -- | -- |
| vHeat-T| 512x512 | 62M | 948G | UperNet@160k | 46.9| [config](segmentation/configs/vheat/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py)/[log](#)/[ckpt](#) |
| Swin-S | 512x512 | 81M | 1039G | UperNet@160k | 47.6| -- | -- |
| vHeat-S| 512x512 | 82M | 1028G | UperNet@160k | 49.0|[config](segmentation/configs/vheat/upernet_vssm_4xb4-160k_ade20k-512x512_small.py)/[log](#)/[ckpt](#) |
| Swin-B | 512x512 | 121M | 1188G | UperNet@160k | 48.1|-- |
| vHeat-B| 512x512 | 129M | 1219G | UperNet@160k | 49.6|[config](segmentation/configs/vheat/upernet_vssm_4xb4-160k_ade20k-512x512_base.py)/[log](#)/[ckpt](#) |


* *Models in this subsection is initialized from the models trained in `classfication`.*

## Getting Started
### Installation

**Step 1: Clone the vHeat repository:**

To get started, first clone the vHaet repository and navigate to the project directory:

```bash
git clone https://github.com/MzeroMiko/vHeat.git
cd vHeat
```

**Step 2: Environment Setup:**

***Create and activate a new conda environment***

```bash
conda create -n vHeat
conda activate vHeat
```

***Install Dependencies***

```bash
pip install -r requirements.txt
```

***Dependencies for `Detection` and `Segmentation` (optional)***

```bash
pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless ftfy regex
pip install mmdet==3.3.0 mmsegmentation==1.2.2 mmpretrain==1.2.0
```


### Model Training and Inference

**Classification**

To train vHeat models for classification on ImageNet, use the following commands for different configurations:

```bash
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=16 --master_addr="127.0.0.1" --master_port=29501 main.py --cfg </path/to/config> --batch-size 128 --data-path </path/to/dataset> --output /tmp
```

If you only want to test the performance (together with params and FLOPs):

```bash
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29501 main.py --cfg </path/to/config> --batch-size 128 --data-path </path/to/dataset> --output /tmp --resume </path/to/checkpoint> --eval --model_ema False
```

***please refer to [modelcard](./modelcard.sh) for more details.***

**Detection and Segmentation**

To evaluate with `mmdetection` or `mmsegmentation`:
```bash
bash ./tools/dist_test.sh </path/to/config> </path/to/checkpoint> 1
```
*use `--tta` to get the `mIoU(ms)` in segmentation*

To train with `mmdetection` or `mmsegmentation`:
```bash
bash ./tools/dist_train.sh </path/to/config> 8
```

For more information about detection and segmentation tasks, please refer to the manual of [`mmdetection`](https://mmdetection.readthedocs.io/en/latest/user_guides/train.html) and [`mmsegmentation`](https://mmsegmentation.readthedocs.io/en/latest/user_guides/4_train_test.html). Remember to use the appropriate backbone configurations in the `configs` directory.

***Before training on downstream tasks (detection/segmentation), please run [interpolate4downstream.py](classification/interpolate4downstream.py) to modify the classification pre-trained checkpoint to load for training.***



