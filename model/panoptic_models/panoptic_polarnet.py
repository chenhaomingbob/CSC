# from pcseg.models.detectors.detector3d_template import Detector3DTemplate
# from pcseg.models.loss import build_losses
# from pcseg.models.loss.lovasz_losses import lovasz_softmax


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# from .sub_layers import cylinder_fea, Asymm_3d_spconv, BEV_Unet

from model.panoptic_models.eval_miou import fast_hist_crop
from model.panoptic_models.post_processing import get_panoptic_segmentation
from model.panoptic_models.lovasz_losses import lovasz_softmax

from model.cylinder3d.cylinder3D import Cylinder3D
from model.panoptic_models.BEV_Unet import UNet
from model.panoptic_models.sub_layers import Asymm_3d_spconv
import torch_scatter
import spconv.pytorch as spconv


class Panoptic_PolarNet(nn.Module):
    def __init__(self, config):
        super(Panoptic_PolarNet, self).__init__()

        self.grid_size = [480, 360, 32]  #
        # self.num_classes = 20
        self.output_dim_3d_backbone = config['output_dim_model_points']  # 20 for nuScenes
        # num_thing_class = len([1, 2, 3, 4, 5, 6, 7, 8])  # i n semantickitti dataset

        n_class = config['model_n_out']
        n_input = 64
        dilation = 1
        dropout = 0.5

        self.model_point_name = config['model_points'].lower()  #
        if self.model_point_name == 'minkunet':
            raise Exception("Not implementation")
        elif self.model_point_name == 'voxelnet':
            raise Exception("Not implementation")
        elif self.model_point_name in ('cylinder3d', 'cylinder3d_separate'):
            self.backbone_3d = Cylinder3D(input_dim=4, output_dim=self.output_dim_3d_backbone, cfg=config)

            self.voxel_sem_head = spconv.SubMConv3d(
                self.output_dim_3d_backbone,
                n_class,
                indice_key="voxel_sem",
                kernel_size=3, stride=1,
                padding=1,
                bias=True
            )

        else:
            raise Exception("Unknown 3d backbone")

        self.backbone_2d = UNet(n_class=n_class, n_height=n_input, dilation=dilation, group_conv=False,
                                input_batch_norm=True, dropout=dropout, circular_padding=False, dropblock=True)

        ignore_label = 255
        self.ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)
        self.center_loss_fn = torch.nn.MSELoss()
        self.offset_loss_fn = torch.nn.L1Loss()
        self.center_loss_weight = 100
        self.offset_loss_weight = 10

    def forward(self, batch_dict):
        # batch_size = batch_dict['batch_size']

        voxel_features, bev_features, voxel_sem_prediction = None, None, None
        if self.model_point_name == 'minkunet':
            raise Exception("Not implementation")
        elif self.model_point_name == 'voxelnet':
            raise Exception("Not implementation")
        elif self.model_point_name in ('cylinder3d', 'cylinder3d_separate'):
            voxel_features, bev_features = self.backbone_3d(batch_dict)
            voxel_sem_prediction = self.voxel_sem_head(voxel_features)
            voxel_sem_prediction = voxel_sem_prediction.dense()  # (B,C,H,W,D)  e.g., (4,20,480,360,32)
        ##############################
        bev_features = bev_features.dense().squeeze(-1)  # (B,C,H,W,1) -> (B,C,H,W)

        center, offset = self.backbone_2d(bev_features)
        # center: (4,1,480,360) ; offset: (4, 2, 480, 360)
        output_dict = {
            'voxel_prediction': voxel_sem_prediction,  # (4, 20, 480, 360, 32)
            'pred_center': center,  # (4, 1, 480, 360)
            'pred_offset': offset  # (4, 2, 480, 360)
        }
        return output_dict

        # if self.training:
        #     total_loss, loss_dict = self.get_training_loss(output_dict, batch_dict)
        #     return total_loss
        # else:
        #     pred_dicts, recall_dicts = self.post_processing(output_dict, batch_dict)
        #     return pred_dicts, recall_dicts

        #####
        # # pt_fea: [(122078,9)].  xy_ind_tensor: [(122078,3)]
        # pt_fea, xy_ind_tensor = batch_dict["point_feat"], batch_dict['point_voxel_index']
        #
        # # 对每个网格的点进行池化
        # coords, features_3d = self.cylinder_3d_generator(pt_fea, xy_ind_tensor, batch_dict)
        # # 主干网络
        # sem_prediction, center_prediction, offset_prediction = self.cylinder_3d_spconv_seg(features_3d, coords,
        #                                                                                    len(pt_fea), batch_dict,
        #                                                                                    debug=False)
        # # 稠密heatmap网络
        # center_prediction = self.UNet(center_prediction)
        #
        # output_dict = {
        #     'voxel_prediction': sem_prediction,
        #     'pred_center': center_prediction,
        #     'pred_offset': offset_prediction
        # }
        # # output_dict.update(debug_tensor)
        # if self.training:
        #     total_loss, loss_dict = self.get_training_loss(output_dict, batch_dict)
        #     return total_loss, loss_dict, output_dict
        # else:
        #     pred_dicts, recall_dicts = self.post_processing(output_dict, batch_dict)
        #     return pred_dicts, recall_dicts

    def post_processing(self, output_dict, batch_dict):

        recall_dicts = {}

        # GT data
        gt_points_grid_index = [x.cpu().numpy() for x in batch_dict['point_voxel_index']]
        gt_points_label = batch_dict['point_sem_label']
        gt_points_instance = batch_dict['point_inst_label']
        gt_voxel_sem = batch_dict['voxel_sem_label']
        unique_label = batch_dict['meta'][0]['unique_label']
        gt_center = batch_dict['gt_center']
        gt_offset = batch_dict['gt_offset']

        # prediction
        voxel_prediction = output_dict['voxel_prediction']  # (B,20,480,360,32)
        pred_center = output_dict['pred_center']
        pred_offset = output_dict['pred_offset']

        origin_thing_list = [1, 2, 3, 4, 5, 6, 7, 8]
        processed_thing_list = [x - 1 for x in origin_thing_list]  # [0,1,2,3,4,5,6,7]
        #
        post_proc_threshold, post_proc_nms_kernel, post_proc_top_k, post_proc_use_polar = 0.1, 5, 100, True
        use_pred = True
        #
        voxel_panoptic_results, point_panoptic_results, point_gt_labels_list, point_gt_instances_list = [], [], [], []
        sem_hist_list = []
        for cur_sample_index, cur_gt_points_grid_index in enumerate(gt_points_grid_index):
            ###### Panoptic Segmentation ######
            # get foreground_mask
            for_mask = torch.zeros(1, self.grid_size[0], self.grid_size[1], self.grid_size[2], dtype=torch.bool).cuda()
            for_mask[0, cur_gt_points_grid_index[:, 0], cur_gt_points_grid_index[:, 1], cur_gt_points_grid_index[:,
                                                                                        2]] = True
            ##
            if use_pred:
                cur_voxel_prediction = voxel_prediction[cur_sample_index]  # 要求数值的范围在 [0~20]
                cur_center, cur_offset = pred_center[cur_sample_index], pred_offset[cur_sample_index]

                cur_voxel_panoptic, center_points = get_panoptic_segmentation(
                    torch.unsqueeze(cur_voxel_prediction, 0), torch.unsqueeze(cur_center, 0),
                    torch.unsqueeze(cur_offset, 0), origin_thing_list,
                    threshold=post_proc_threshold, nms_kernel=post_proc_nms_kernel, top_k=post_proc_top_k,
                    polar=post_proc_use_polar, foreground_mask=for_mask)

            else:
                cur_voxel_prediction = gt_voxel_sem[cur_sample_index]  # (480,360,32) 要求数值的范围在 [0~20]
                cur_center, cur_offset = gt_center[cur_sample_index], gt_offset[
                    cur_sample_index]  # (1,480,360) (2,480,360)

                cur_voxel_panoptic, center_points = get_panoptic_segmentation(
                    torch.unsqueeze(cur_voxel_prediction, 0), torch.unsqueeze(cur_center, 0),
                    torch.unsqueeze(cur_offset, 0), origin_thing_list,
                    threshold=post_proc_threshold, nms_kernel=post_proc_nms_kernel, top_k=post_proc_top_k,
                    polar=post_proc_use_polar, foreground_mask=for_mask)

            cur_voxel_panoptic = cur_voxel_panoptic.cpu().detach().numpy().astype(np.int32)
            point_panoptic = cur_voxel_panoptic[
                0, cur_gt_points_grid_index[:, 0], cur_gt_points_grid_index[:, 1], cur_gt_points_grid_index[:, 2]]

            voxel_panoptic_results.append(cur_voxel_panoptic)
            point_panoptic_results.append(point_panoptic)
            point_gt_labels_list.append(np.squeeze(gt_points_label[cur_sample_index]))  # point的 gt class id
            point_gt_instances_list.append(np.squeeze(gt_points_instance[cur_sample_index]))  # point的 gt instance id

            ###### Semantic Segmentation #######
            if use_pred:
                # semantic segmentation
                cur_voxel_semantic = torch.argmax(cur_voxel_prediction, dim=0)  # [0~19] (20,480,360,32) -> (480,360,32)
                # shift back to original label idx
                cur_voxel_semantic = torch.add(cur_voxel_semantic, 1).cpu().detach().numpy().astype(
                    np.int32)  # [0~19] -> [1~20]
            else:
                # semantic segmentation
                cur_voxel_semantic = cur_voxel_prediction.type(torch.ByteTensor)  # [0~19,255] (480,360,32)
                # shift back to original label idx
                cur_voxel_semantic = torch.add(cur_voxel_semantic, 1).type(
                    torch.LongTensor).cpu().detach().numpy().astype(np.int32)  # [0~19,255] -> [0~20]

            sem_hist_list.append(
                fast_hist_crop(output=cur_voxel_semantic[
                    cur_gt_points_grid_index[:, 0], cur_gt_points_grid_index[:, 1], cur_gt_points_grid_index[:, 2]],
                               target=gt_points_label[cur_sample_index], unique_label=unique_label))

        panoptic_outputs = {
            'voxel_pred_panoptic_list': voxel_panoptic_results,
            'point_pred_panoptic_list': point_panoptic_results,
            'point_gt_label_list': point_gt_labels_list,
            'point_gt_instance_list': point_gt_instances_list,
            'sem_hist_list': sem_hist_list
        }
        output_dict.update(panoptic_outputs)
        return output_dict, recall_dicts

    def get_training_loss(self, output_dict, batch_dict):
        # predictions of network
        pred_voxel_sem = output_dict['voxel_prediction']  # (B, 20, 480, 360, 32)
        pred_center = output_dict['pred_center']  # (B, 1, 480, 360)
        pred_offset = output_dict['pred_offset']  # (B, 2, 480, 360)

        # ground truth
        gt_voxel_sem = batch_dict['voxel_sem_label']  # (B, 480, 360, 32)
        gt_center = batch_dict['gt_center']  # (B, 1, 480, 360)
        gt_offset = batch_dict['gt_offset']  # (B, 2, 480, 360)

        total_loss, loss_dict = self.cal_panoptic_loss(
            sem_prediction=pred_voxel_sem, center_prediction=pred_center, offset_prediction=pred_offset,
            voxel_label=gt_voxel_sem, center_gt=gt_center, offset_gt=gt_offset,
            output_dict=output_dict
        )

        return total_loss, loss_dict

    def cal_panoptic_loss(self, sem_prediction, center_prediction, offset_prediction, voxel_label, center_gt, offset_gt,
                          output_dict=None):
        loss_dict = {}

        # semantic loss
        # sem_prediction = sem_prediction + 1e-5  # range of float16 is 5.960464477539063e-08 ~ 65504

        ce_loss = self.ce_loss_fn(sem_prediction, voxel_label)
        lz_loss = lovasz_softmax(torch.nn.functional.softmax(sem_prediction, dim=1), voxel_label, ignore=255)
        loss = lz_loss + ce_loss

        loss_dict['LovaszSoftmaxLoss'] = lz_loss.cpu().detach().item()
        loss_dict['CrossEntropyLoss'] = ce_loss.cpu().detach().item()
        # center heatmap loss
        center_mask = (center_gt > 0.01) | (torch.min(torch.unsqueeze(voxel_label, 1), dim=4)[0] < 255)
        view = center_gt.cpu().numpy()
        center_loss = self.center_loss_fn(center_prediction, center_gt) * center_mask
        # safe division
        if center_mask.sum() > 0:
            center_loss = center_loss.sum() / center_mask.sum() * self.center_loss_weight
        else:
            center_loss = center_loss.sum() * 0
        loss_dict['InstanceCenterHeatmapLoss'] = center_loss.cpu().detach().item()

        loss += center_loss
        # offset loss
        offset_mask = offset_gt != 0
        offset_loss = self.offset_loss_fn(offset_prediction, offset_gt) * offset_mask
        # safe division
        if offset_mask.sum() > 0:
            offset_loss = offset_loss.sum() / offset_mask.sum() * self.offset_loss_weight
        else:
            offset_loss = offset_loss.sum() * 0

        loss_dict['InstanceOffsetLoss'] = offset_loss.cpu().detach().item()
        loss += offset_loss
        loss_dict['loss'] = loss.cpu().detach().item()

        return loss, loss_dict
