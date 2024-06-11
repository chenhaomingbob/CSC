# Building a Strong Pre-Training Baseline for Universal 3D Large-Scale Perception

This is official implementation of our CVPR 2024 paper "Building a Strong Pre-Training Baseline for Universal 3D
Large-Scale Perception"

[paper](https://arxiv.org/abs/2405.07201)

# Updates

- [2024-05-15] We are preparing the code and expect to release it before June 19.
- [2024-06-11] Initialize the release code.

# Requirements

- python=3.9
- pytorch=2.1.0
- lightning=2.1.0

```shell
conda create -n py39_pyt210_cu118 python==3.9 -y
conda activate py39_pyt210_cu118

# install pytorch==2.1.0
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
or
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# install MinkowskiEngine following https://github.com/NVIDIA/MinkowskiEngine
# for example:
pip install ninja
git clone https://github.com/NVIDIA/MinkowskiEngine
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas

# install torch-scatter
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

pip install -r requirements.txt
```

# Datasets

1. Download nuScenes dataset from the official [link](https://www.nuscenes.org/nuscenes#overview), and put the dataset
   in `{project_root}/datasets/nuscenes`
2. Download the superpixels `superpixels_dinov2_ade20k.zip` from
   the [BAIDU](https://pan.baidu.com/s/1WavtRbc5tGUvsOGtRDJpxw?pwd=6tlq), and unzip the file
   under `{project_root}/superpixels/nuscenes`

the project structure should be like:

```shell
{project_root}
|--config
|--downstream
|--model
|--pretrain
|--utils
|--datasets
  |--nuscenes
    |--samples
    |--sweeps
    |--lidarseg
    |--nuScenes-panoptic-v1.0-all
    |--v1.0-trainval
|--superpixels
  |--nuscenes
    |--superpixels_dinov2_ade20k
|--...
```

# Experiments

## 3D Semantic Segmentation

```shell
# 1. pre-train the 3d backbone MinkUNet
CUDA_VISIBLE_DEVICES=0,1 python pretrain_cluster_prototype.py --cfg config/pretrain/csc_minkunet_dinov2_g2b16.yaml
# the {pretrain_weights_path}   will be found in `{project_root}/output/pretrain/nuscenes/cp/v1_1/{year}_{month}_{day}_{hour}_{minute}/final_model_cp_v1_1.pt`

# 2. fine-tune the 3d backbone using our provided script
sh downstream_semseg_finetune.sh 0,1 {pretrain_weights_path} csc_sem_seg
```

## 3D Object Detection

```shell
#1. pre-train the 3D backbone VoxelNet
CUDA_VISIBLE_DEVICES=0,1 python pretrain_cluster_prototype.py --cfg  config/pretrain/csc_voxelnet_dinov2_g2b16.yaml

#2. fine-tune the VoxelNet using OpenPCDet, https://github.com/open-mmlab/OpenPCDet. 
# Please refer to the TriCC https://openaccess.thecvf.com/content/CVPR2023/html/Pang_Unsupervised_3D_Point_Cloud_Representation_Learning_by_Triangle_Constrained_Contrast_CVPR_2023_paper.html
```

## 3D Panoptic Segmentation

```shell
# 1. pre-train the 3d backbone Cylinder3D
CUDA_VISIBLE_DEVICES=0,1 python  pretrain_cluster_prototype.py --cfg_file config/pretrain/csc_cylinder3d_dinov2_g2b16.yaml
# the pre-training weights will be found in `{project_root}/output/pretrain/cp_V1_1/panoptic_polarnet_cylinder3d/dinov2_ade20k/{year}_{month}_{day}_{hour}_{minute}/model.pt`

# 2. fine-tune the 3d backbone using our provided script
sh downstream_panseg_finetune.sh 0,1 {pretrain_weights_path} csc_pan_seg
```

# Acknowledgement

The codebase is adapted from [SLidR](https://github.com/valeoai/SLidR).

# Citation
```
@InProceedings{chen2024building,
title={Building a Strong Pre-Training Baseline for Universal 3D Large-Scale Perception},
author={Chen, Haoming and Zhang, Zhizhong and Qu, Yanyun and Zhang, Ruixin and Tan, Xin and Xie, Yuan},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year= {2024}
}
```





