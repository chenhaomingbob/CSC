#!/bin/bash
# 使用范例:
# sh sem_downstream.sh 0,1 model.pt

# 获取参数
GPU_IDS=$1
MODEL_PATH=$2
EXP_NAME=$3

export CUDA_VISIBLE_DEVICES=$GPU_IDS

################### nuScenes ###################
# 1% fine-tuning
echo "Starting 1% label fine-tuning in nuScenes dataset"
echo "python downstream_panoptic.py --cfg_file ./config/downstream/panseg/panoptic_polarnet/fintune/panseg_nuscenes_cylinder3d_1pt_g2b8_pure.yaml --pretraining_path $MODEL_PATH --exp_name $EXP_NAME"
python downstream_panoptic.py --cfg_file ./config/downstream/panseg/panoptic_polarnet/fintune/panseg_nuscenes_cylinder3d_1pt_g2b8_pure.yaml --pretraining_path $MODEL_PATH --exp_name $EXP_NAME

# 5% fine-tuning
echo "Starting 5% label fine-tuning in nuScenes dataset"
echo "python downstream_panoptic.py --cfg_file ./config/downstream/panseg/panoptic_polarnet/fintune/panseg_nuscenes_cylinder3d_5pt_g2b8_pure.yaml  --pretraining_path $MODEL_PATH --exp_name $EXP_NAME"
python downstream_panoptic.py --cfg_file ./config/downstream/panseg/panoptic_polarnet/fintune/panseg_nuscenes_cylinder3d_5pt_g2b8_pure.yaml --pretraining_path $MODEL_PATH --exp_name $EXP_NAME
# 10% fine-tuning
echo "Starting 10% label fine-tuning in nuScenes dataset"
echo "python downstream_panoptic.py --cfg_file ./config/downstream/panseg/panoptic_polarnet/fintune/panseg_nuscenes_cylinder3d_10pt_g2b8_pure.yaml  --pretraining_path $MODEL_PATH --exp_name $EXP_NAME"
python downstream_panoptic.py --cfg_file ./config/downstream/panseg/panoptic_polarnet/fintune/panseg_nuscenes_cylinder3d_10pt_g2b8_pure.yaml --pretraining_path $MODEL_PATH --exp_name $EXP_NAME

# 25% fine-tuning
echo "Starting 25% label fine-tuning in nuScenes dataset"
echo "python downstream_panoptic.py --cfg_file ./config/downstream/panseg/panoptic_polarnet/fintune/panseg_nuscenes_cylinder3d_25pt_g2b8_pure.yaml  --pretraining_path $MODEL_PATH --exp_name $EXP_NAME"
python downstream_panoptic.py --cfg_file ./config/downstream/panseg/panoptic_polarnet/fintune/panseg_nuscenes_cylinder3d_25pt_g2b8_pure.yaml --pretraining_path $MODEL_PATH --exp_name $EXP_NAME
# 100% fine-tuning
echo "Starting 100% label fine-tuning in nuScenes dataset"
echo "python downstream_panoptic.py --cfg_file ./config/downstream/panseg/panoptic_polarnet/fintune/panseg_nuscenes_cylinder3d_100pt_g2b8_pure.yaml  --pretraining_path $MODEL_PATH --exp_name $EXP_NAME"
python downstream_panoptic.py --cfg_file ./config/downstream/panseg/panoptic_polarnet/fintune/panseg_nuscenes_cylinder3d_100pt_g2b8_pure.yaml --pretraining_path $MODEL_PATH --exp_name $EXP_NAME
##################################################
exit 0
