import logging
import os.path
from copy import deepcopy

import numpy as np
import torch
from MinkowskiEngine import SparseTensor
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from pretrain.prototype.slotcon import DINOHead
from utils.metrics import compute_IoU
import json

CLASSES_NUSCENES = [
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
]

CLASSES_KITTI = [
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]


def evaluate_semseg(model, dataloader, config):
    """
    Function to evaluate the performances of a downstream training.
    It prints the per-class IoU, mIoU and fwIoU.
    """
    model.eval()
    model_point_backbone = config['backbone']
    working_dir = config['working_dir']
    ignore_index = config['ignore_index']
    prediction_save_dir = os.path.join(working_dir, 'predictions')
    os.makedirs(prediction_save_dir, exist_ok=True)
    with torch.no_grad():
        i = 0
        full_predictions = []
        ground_truth = []
        for batch in tqdm(dataloader, dynamic_ncols=True):
            if model_point_backbone == 'MinkUnet18A'.lower():
                batch["sinput_F"] = torch.ones((batch["sinput_F"].shape[0], 3), device=batch["sinput_F"].device)
            lidar_tokens = batch['lidar_tokens']
            sparse_input = SparseTensor(batch["sinput_F"], batch["sinput_C"], device=0)
            output_points = model(sparse_input).F
            if ignore_index:
                output_points[:, ignore_index] = -1e6

            torch.cuda.empty_cache()
            preds = output_points.argmax(1).cpu()
            offset = 0
            for j, lb in enumerate(batch["len_batch"]):
                inverse_indexes = batch["inverse_indexes"][j]
                predictions = preds[inverse_indexes + offset]

                # remove the ignored index entirely
                full_predictions.append(predictions)
                ground_truth.append(deepcopy(batch["evaluation_labels"][j]))
                offset += lb
                ################
                ###  保存结果  ###
                ################
                token = lidar_tokens[j]
                save_path = os.path.join(prediction_save_dir, f"{token}.npy")
                np.save(save_path, predictions)
            i += j
        m_IoU, fw_IoU, per_class_IoU = compute_IoU(
            torch.cat(full_predictions),
            torch.cat(ground_truth),
            config["model_n_out"],
            ignore_index=0,
        )

        print("Per class IoU:")
        per_class_value = {}
        if config["dataset"].lower() == "nuscenes":
            for a, b in zip(CLASSES_NUSCENES, (per_class_IoU).numpy()):
                print(f"{a:20} - {b:.3f}")
                per_class_value[str(a)] = format(b, '.4f')
        elif config["dataset"].lower() == "kitti":
            for a, b in zip(CLASSES_KITTI, (per_class_IoU).numpy()):
                print(f"{a:20} - {b:.3f}")
                per_class_value[str(a)] = format(b, '.4f')

        print(f"mIoU: {m_IoU}")
        print(f"fwIoU: {fw_IoU}")

        results_dict = {
            "mIoU": m_IoU,
            "fwIoU": fw_IoU,
            "per_class_value": per_class_value
        }
        results_dict = json.dumps(results_dict, sort_keys=True)
        with open(os.path.join(working_dir, 'evaluation_metrics.json'), 'w') as f:
            f.write(results_dict)

    return m_IoU


def inference_similarity(module, dataloader, config, **kwargs):
    """
       Function to evaluate the performances of a downstream training.
       It prints the per-class IoU, mIoU and fwIoU.
    """
    module.switch_model()
    working_dir = config['working_dir']

    # ignore_index = config['ignore_index']
    dataset = config['dataset'].lower()
    assert dataset in ['nuscenes']
    mixed_prototype_save_path = os.path.join(working_dir, f"mixed_prototype.npy")
    np.save(
        file=mixed_prototype_save_path,
        arr=module.get_mixed_prototype().cpu().detach().numpy()
    )
    prediction_save_dir = os.path.join(working_dir, 'predictions')
    os.makedirs(prediction_save_dir, exist_ok=True)
    print(f"working_dir:{os.path.abspath(working_dir)}\n"
          f"prediction_save_dir:{os.path.abspath(prediction_save_dir)}")

    with torch.no_grad():
        for batch in tqdm(dataloader, dynamic_ncols=True):
            output_dict = module.inference(batch)
            ######################################################
            pc_xyz_list = batch['sinput_C']
            pred_voxels = output_dict['pred_voxels']  #
            pred_points = output_dict['pred_points']
            evaluation_point_labels = batch['evaluation_point_labels']
            evaluation_voxel_labels = batch['evaluation_voxel_labels']
            valid_superpoint_feats = output_dict['valid_superpoint_feats']
            valid_superpixel_feats = output_dict['valid_superpixel_feats']
            image_lists = output_dict['image_list']
            ######################################################
            lidar_tokens = batch['lidar_tokens']
            for i in range(len(pred_voxels)):
                ###
                pc_xyz = pc_xyz_list[pc_xyz_list[:, 0] == i][:, 1:]
                pred_voxel = pred_voxels[i]  # (N_{voxel},64)
                pred_point = pred_points[i]  # (N_{point},64)
                gt_point = evaluation_point_labels[i]
                gt_voxel = evaluation_voxel_labels[i]
                valid_superpoint_feat = valid_superpoint_feats[i]
                valid_superpixel_feat = valid_superpixel_feats[i]
                lidar_token = lidar_tokens[i]
                cur_image_list = []
                for j in range(6):
                    cur_image_list.append(image_lists[i * 6 + j])
                ###
                token = lidar_tokens[i]['LIDAR_TOP']
                save_path = os.path.join(prediction_save_dir, f"{token}.npy")
                ###################
                np.save(
                    file=save_path, arr={
                        'pc_xyz': pc_xyz,
                        'pred_voxel': pred_voxel,
                        'pred_point': pred_point,
                        'gt_point': gt_point,
                        'gt_voxel': gt_voxel,
                        'valid_superpoint_feat': valid_superpoint_feat,
                        'valid_superpixel_feat': valid_superpixel_feat,
                        'image_list': cur_image_list,
                        'lidar_token': lidar_token,
                    }
                )


def unsup_evaluate(model, dataloader, config):
    """
    无监督语义分割效果
    """
    model.eval()
    logger = logging.getLogger(__name__)

    # point_prototype = SemanticGrouping(num_slots=16, dim_slot=64)
    pretraining_file = config['pretraining_path']
    checkpoint = torch.load(pretraining_file, map_location='cpu')
    # pixel_prototype
    pixel_prototype = checkpoint['pixel_prototype']
    point_prototype = checkpoint['point_prototype']
    # point_prototype_weight = checkpoint['model_grouping_point']['slot_embed.weight']
    predictor_slot = DINOHead(64, hidden_dim=128, bottleneck_dim=64)
    predictor_slot.load_state_dict(checkpoint['model_predictor_slot'])

    all_preds, all_label = [], []
    with torch.no_grad():
        i = 0
        full_predictions = []
        ground_truth = []
        for batch in tqdm(dataloader):
            if '3d_model' in config and config['3d_model'] == 'MinkUnet18A':
                batch["sinput_F"] = torch.ones((batch["sinput_F"].shape[0], 3), device=batch["sinput_F"].device)
            sparse_input = SparseTensor(batch["sinput_F"], batch["sinput_C"], device=0)
            output_points = model(sparse_input).F
            if config["ignore_index"]:
                output_points[:, config["ignore_index"]] = -1e6

            torch.cuda.empty_cache()
            preds = output_points.argmax(1).cpu()
            offset = 0
            for j, lb in enumerate(batch["len_batch"]):
                inverse_indexes = batch["inverse_indexes"][j]
                predictions = preds[inverse_indexes + offset]

                # remove the ignored index entirely
                full_predictions.append(predictions)
                ground_truth.append(deepcopy(batch["evaluation_labels"][j]))
                offset += lb
            i += j

    m_IoU, fw_IoU, per_class_IoU = compute_IoU(
        torch.cat(full_predictions),
        torch.cat(ground_truth),
        config["model_n_out"],
        ignore_index=0,
    )

    all_preds = torch.cat(full_predictions).numpy()
    all_labels = torch.cat(ground_truth).numpy()
    ignore_index = 0
    '''Unsupervised, Match pred to gt'''
    sem_num = config["model_n_out"]  # 17 for nuscenes
    mask = (all_labels >= 0) & (all_labels < sem_num)
    histogram = np.bincount(sem_num * all_labels[mask] + all_preds[mask], minlength=sem_num ** 2).reshape(sem_num,
                                                                                                          sem_num)
    # histogram[ignore_index] =0.0
    '''Hungarian Matching'''
    indices = linear_sum_assignment(histogram.max() - histogram)
    m = np.transpose(np.asarray(indices))

    o_Acc = histogram[m[:, 0], m[:, 1]].sum() / histogram.sum() * 100.
    m_Acc = np.mean(histogram[m[:, 0], m[:, 1]] / histogram.sum(1)) * 100
    hist_new = np.zeros((sem_num, sem_num))
    for idx in range(sem_num):
        hist_new[:, idx] = histogram[:, m[idx, 1]]

    '''Final Metrics'''
    # hist_new[ignore_index]=0
    tp = np.diag(hist_new)  # 取对角
    fp = np.sum(hist_new, axis=0) - tp  # (17,17) -> (1,17) -> (17)
    fn = np.sum(hist_new, axis=1) - tp  # (17,17) -> (17,1) -> (17)
    IoUs = tp / (tp + fp + fn + 1e-8)
    m_IoU = np.nanmean(IoUs)
    s = '| mIoU {:5.2f} | \n'.format(100 * m_IoU)

    for index, IoU in enumerate(IoUs):
        if index == 0:
            s += '{} :{:5.2f}\n'.format("ignore".center(20), 100 * IoU)
        else:
            s += '{} :{:5.2f}\n'.format(CLASSES_NUSCENES[index - 1].center(20), 100 * IoU)

    logger.info(f"o_Acc: {o_Acc}")
    logger.info(f"m_Acc: {m_Acc}")
    logger.info(f"IoU: {s}")
    print(f"o_Acc: {o_Acc}")
    print(f"m_Acc: {m_Acc}")
    print(f"IoU: {s}")

    # print("Per class IoU:")
    # logger.info("Per class IoU:")
    # if config["dataset"].lower() == "nuscenes":
    #     for a, b in zip(CLASSES_NUSCENES, (per_class_IoU).numpy()):
    #         logger.info(f"{a:20} - {b:.3f}")
    # elif config["dataset"].lower() == "kitti":
    #     for a, b in zip(CLASSES_KITTI, (per_class_IoU).numpy()):
    #         logger.info(f"{a:20} - {b:.3f}")
    #
    # logger.info(f"mIoU: {m_IoU}")
    # logger.info(f"fwIoU: {fw_IoU}")
    # print(f"mIoU: {m_IoU}")
    # print(f"fwIoU: {fw_IoU}")

    return m_IoU


import torch.nn as nn
import torch.functional as F


class SemanticGrouping(nn.Module):
    def __init__(self, num_slots, dim_slot, temp=0.07, eps=1e-6):
        super().__init__()
        self.num_slots = num_slots
        self.dim_slot = dim_slot
        self.temp = temp
        self.eps = eps

        self.slot_embed = nn.Embedding(num_slots, dim_slot)  #
        ## debug ##
        self.unique_list = {}

    def forward(self, x):
        x = torch.permute(x, (1, 0))  # (N,64) -> (64,N)
        x_prev = x  # (d,n)/(64,1793)
        slots = self.slot_embed(torch.arange(0, self.num_slots, device=x.device))  # (k,d) / (32,64)
        dots = torch.einsum('kd,dn -> kn',
                            F.normalize(slots, dim=1),
                            F.normalize(x, dim=0))  # (k,n) / (32,1793)
        attn = (dots / self.temp).softmax(dim=0) + self.eps  # (k,n) / (32,1793)
        ### 查看激活的情况
        # (1,1793)
        actived_slots, counts = torch.unique(torch.argmax(attn, dim=0, keepdim=True), return_counts=True)
        for a, b in zip(actived_slots, counts):
            a = int(a.detach().cpu())
            b = int(b.detach().cpu())
            if a not in self.unique_list:
                self.unique_list[a] = b
            else:
                self.unique_list[a] += b

        #

        slots = torch.einsum('dn,kn->kd',
                             x_prev,
                             attn / attn.sum(dim=1, keepdim=True))  # (k,d) / (32,64)
        return slots, dots  # (k,d) (k,n) / (32,64) (32,1793)
