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
echo "Starting 1% label fine-tuning in nuScenes"
echo "python downstream_semantic.py --cfg_file ./config/downstream/semseg/finetune/1pt/semseg_nuscenes_1pt_g2b32.yaml --pretraining_path $MODEL_PATH --exp_name $EXP_NAME"
python downstream_semantic.py --cfg_file ./config/downstream/semseg/finetune/1pt/semseg_nuscenes_1pt_g2b32.yaml --pretraining_path $MODEL_PATH --exp_name $EXP_NAME
# 5% fine-tuning
echo "Starting 5% label fine-tuning in nuScenes"
echo "python downstream_semantic.py --cfg_file ./config/downstream/semseg/finetune/5pt/semseg_nuscenes_5pt_g2b32.yaml --pretraining_path $MODEL_PATH --exp_name $EXP_NAME"
python downstream_semantic.py --cfg_file ./config/downstream/semseg/finetune/5pt/semseg_nuscenes_5pt_g2b32.yaml --pretraining_path $MODEL_PATH --exp_name $EXP_NAME
# 10% fine-tuning
echo "Starting 10% label fine-tuning in nuScenes"
echo "python downstream_semantic.py --cfg_file ./config/downstream/semseg/finetune/10pt/semseg_nuscenes_10pt_g2b32.yaml --pretraining_path $MODEL_PATH --exp_name $EXP_NAME"
python downstream_semantic.py --cfg_file ./config/downstream/semseg/finetune/10pt/semseg_nuscenes_10pt_g2b32.yaml --pretraining_path $MODEL_PATH --exp_name $EXP_NAME
# 25% fine-tuning
echo "Starting 25% label fine-tuning in nuScenes"
echo "python downstream_semantic.py --cfg_file ./config/downstream/semseg/finetune/25pt/semseg_nuscenes_25pt_g2b32.yaml --pretraining_path $MODEL_PATH --exp_name $EXP_NAME"
python downstream_semantic.py --cfg_file ./config/downstream/semseg/finetune/25pt/semseg_nuscenes_25pt_g2b32.yaml --pretraining_path $MODEL_PATH --exp_name $EXP_NAME
# 100% fine-tuning
echo "Starting 100% label fine-tuning in nuScenes"
echo "python downstream_semantic.py --cfg_file ./config/downstream/semseg/finetune/100pt/semseg_nuscenes_100pt_g2b32.yaml --pretraining_path $MODEL_PATH --exp_name $EXP_NAME"
python downstream_semantic.py --cfg_file ./config/downstream/semseg/finetune/100pt/semseg_nuscenes_100pt_g2b32.yaml --pretraining_path $MODEL_PATH --exp_name $EXP_NAME
# 100% linear probing
echo "Starting 100% label linear probing in nuScenes"
echo "python ../downstream_semantic.py --cfg_file ./config/downstream/semseg/lp/semseg_nuscenes_100pt_g2b32.yaml --pretraining_path $MODEL_PATH --exp_name $EXP_NAME"
python downstream_semantic.py --cfg_file ./config/downstream/semseg/lp/semseg_nuscenes_100pt_g2b32.yaml --pretraining_path $MODEL_PATH --exp_name $EXP_NAME
################### SemanticKITTI ###################
# 1% fine-tuning
echo "Starting 1% label fine-tuning in SemanticKITTI"
echo "python ../downstream_semantic.py --cfg_file ./config/downstream/semseg/finetune/1pt/semseg_kitti_1pt_g2b20.yaml --pretraining_path $MODEL_PATH --exp_name $EXP_NAME"
python downstream_semantic.py --cfg_file ./config/downstream/semseg/finetune/1pt/semseg_kitti_1pt_g2b20.yaml --pretraining_path $MODEL_PATH --exp_name $EXP_NAME
####
exit 0
