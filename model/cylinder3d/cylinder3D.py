# -*-coding:utf-8-*-
"""
    Author: chenhaomingbob
    E-mail: chenhaomingbob@163.com
    Time: 2022/05/30
    Description: 
"""
from .lovasz_losses import lovasz_softmax
from .blocks import ResContextBlock, ReconBlock, ResBlock, UpBlock
import spconv.pytorch as spconv
import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def build_loss(wce=True, lovasz=True, num_class=20, ignore_label=0):
    loss_funs = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    if wce and lovasz:
        return loss_funs, lovasz_softmax
    elif wce and not lovasz:
        return wce
    elif not wce and lovasz:
        return lovasz_softmax
    else:
        raise NotImplementedError


class Cylinder3D(nn.Module):
    def __init__(self, input_dim, output_dim, cfg):
        super(Cylinder3D, self).__init__()

        # assert cfg['use_polar'] == True
        self.use_polar = cfg['use_polar']
        dataset_name = cfg["dataset"].lower()

        assert dataset_name == 'nuscenes', "Only support nuscenes dataset"
        if self.use_polar:
            # 极坐标
            self.point_cloud_range = np.array([0, -3.1415926, -3, 50, 3.1415926, 3], dtype=np.float32)
            self.grid_size = [480, 360, 32]
            self.voxel_size = (self.point_cloud_range[3:] - self.point_cloud_range[0:3]) / self.grid_size
        else:
            #
            self.voxel_size = [0.1, 0.1, 0.2]  # nuScenes
            self.point_cloud_range = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], dtype=np.float32)  # nuScenes

        init_size = 32
        num_input_features = 16

        self.input_dim = input_dim
        self.output_dim = output_dim

        ####### Class: cylinder_fea #######
        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(self.input_dim),
            nn.Linear(self.input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 256)
        )

        kernel_size = 3
        self.local_pool_op = nn.MaxPool2d(kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1)

        # point feature compression
        # if self.fea_compre is not None:
        self.fea_compression = nn.Sequential(
            nn.Linear(256, 16),
            nn.ReLU(inplace=True)
        )
        self.pt_fea_dim = 16
        # else:
        #     self.pt_fea_dim = 16
        self.bev_stride = 8
        ####### Class: Asymm_3d_spconv #######
        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5")

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon")

        self.logits = spconv.SubMConv3d(
            4 * init_size,
            self.output_dim,
            indice_key="logit",
            kernel_size=3, stride=1,
            padding=1,
            bias=True
        )
        self.pool3d = spconv.SparseMaxPool3d((1, 1, self.grid_size[-1]), indice_key="pool3d")

    def forward(self, batch_dict):
        pt_fea = batch_dict['pc_features']
        xyz_ind = batch_dict['pc_grid_index']
        batch_size = batch_dict['batch_size']

        cur_dev = pt_fea[0].get_device()
        # concate everything
        cat_pt_ind = []
        for i_batch in range(len(xyz_ind)):
            cat_pt_ind.append(F.pad(xyz_ind[i_batch], (1, 0), 'constant', value=i_batch))

        cat_pt_fea = torch.cat(pt_fea, dim=0)
        cat_pt_ind = torch.cat(cat_pt_ind, dim=0)
        pt_num = cat_pt_ind.shape[0]

        # shuffle the data
        shuffled_ind = torch.randperm(pt_num, device=cur_dev)

        cat_pt_fea = cat_pt_fea[shuffled_ind, :]
        cat_pt_ind = cat_pt_ind[shuffled_ind, :]

        # unique xy grid index
        unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind, return_inverse=True, return_counts=True, dim=0)
        unq = unq.type(torch.int64)

        # process feature
        processed_cat_pt_fea = self.PPmodel(cat_pt_fea)
        pooled_data = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0]

        processed_pooled_data = self.fea_compression(pooled_data)

        coords, voxel_features = unq, processed_pooled_data
        coords = coords.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(
            features=voxel_features,
            indices=coords,
            spatial_shape=self.grid_size,
            batch_size=batch_size
        )
        ret = self.downCntx(ret)
        down1c, down1b = self.resBlock2(ret)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down4c, down4b = self.resBlock5(down3c)

        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet(up1e)

        # up0e.features = torch.cat((up0e.features, up1e.features), 1)
        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))

        voxel_features = self.logits(up0e)
        bev_features = self.pool3d(voxel_features)

        return voxel_features, bev_features

        # return up0e, spconv_tensor
        # N, C, D, H, W = spconv_tensor.shape
        # spconv_tensor = spconv_tensor.view(N, C * D, H, W)  # 变成BEV视角了

        # y = logits.dense()
        # batch_dict['output'] = y

        # return spconv_tensor

        # if self.training:
        #     loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)
        #
        #     ret_dict = {
        #         'loss': loss
        #     }
        #     return ret_dict, tb_dict, disp_dict
        # else:
        #     pred_dicts, recall_dicts = self.post_processing(batch_dict)
        #     return pred_dicts, recall_dicts

    def post_processing(self, batch_dict):
        pred_dicts = {
            "predict_labels": batch_dict["output"]
        }
        recall_dicts = {}
        del batch_dict["output"]
        return pred_dicts, recall_dicts

    def get_training_loss(self, batch_dict):
        disp_dict = {}

        outputs = batch_dict['output']  # [2,2,480,360,32]
        voxel_label = batch_dict['voxel_label']

        loss_lovasz = self.lovasz_softmax(torch.nn.functional.softmax(outputs), voxel_label, ignore=0)
        loss_wce = self.loss_wce(outputs, voxel_label)
        loss = loss_wce + loss_lovasz

        tb_dict = {
            'loss_lovasz': loss_lovasz.item(),
            'loss_wce': loss_wce.item(),
            'loss': loss.item(),
        }

        return loss, tb_dict, disp_dict
