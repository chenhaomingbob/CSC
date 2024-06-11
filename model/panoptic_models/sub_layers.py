# -*-coding:utf-8-*-
"""
    Author: chenhaomingbob
    E-mail: chenhaomingbob@163.com
    Time: 2022/07/02
    Description: 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

import numpy as np
# import spconv
import spconv.pytorch as spconv
import torch
from torch import nn
import math
from dropblock import DropBlock2D


class cylinder_fea(nn.Module):

    def __init__(self, cfgs, grid_size, fea_dim=3,
                 out_pt_fea_dim=64, max_pt_per_encode=64, fea_compre=None):
        super(cylinder_fea, self).__init__()

        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(fea_dim),

            nn.Linear(fea_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, out_pt_fea_dim)
        )

        self.max_pt = max_pt_per_encode
        self.fea_compre = fea_compre
        self.grid_size = grid_size
        kernel_size = 3
        self.local_pool_op = torch.nn.MaxPool2d(kernel_size, stride=1,
                                                padding=(kernel_size - 1) // 2,
                                                dilation=1)
        self.pool_dim = out_pt_fea_dim  # 256

        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                nn.Linear(self.pool_dim, self.fea_compre),
                nn.ReLU(inplace=True))
            self.pt_fea_dim = self.fea_compre
        else:
            self.pt_fea_dim = self.pool_dim

    def forward(self, pt_fea, xy_ind, fusion_dict, return_unq_inv=False):
        cur_dev = pt_fea[0].get_device()

        # concate everything
        cat_pt_ind = []
        for i_batch in range(len(xy_ind)):
            cat_pt_ind.append(F.pad(xy_ind[i_batch], (1, 0), 'constant', value=i_batch))  # 添加 batch_id

        uncat_pt_ind = cat_pt_ind
        cat_pt_fea = torch.cat(pt_fea, dim=0)
        cat_pt_ind = torch.cat(cat_pt_ind, dim=0)
        pt_num = cat_pt_ind.shape[0]

        if 'pairing_points' not in fusion_dict:  # 即
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

        ####################################################
        if self.fea_compre:
            processed_pooled_data = self.fea_compression(pooled_data)
        else:
            processed_pooled_data = pooled_data

        if return_unq_inv:
            return unq, processed_pooled_data, unq_inv
        else:
            return unq, processed_pooled_data


def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


def conv1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride,
                             padding=(0, 1, 1), bias=False, indice_key=indice_key)


def conv1x1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride,
                             padding=(0, 0, 1), bias=False, indice_key=indice_key)


def conv1x3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 1), stride=stride,
                             padding=(0, 1, 0), bias=False, indice_key=indice_key)


def conv3x1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride,
                             padding=(1, 0, 0), bias=False, indice_key=indice_key)


def conv3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 3), stride=stride,
                             padding=(1, 0, 1), bias=False, indice_key=indice_key)


def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ResContextBlock, self).__init__()
        self.conv1 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef1")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.conv1_2 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef2")
        # self.conv1_2 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef")

        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.LeakyReLU(inplace=True)

        self.conv2 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef3")
        self.act2 = nn.LeakyReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef4")
        # self.conv3 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act3 = nn.LeakyReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))

        shortcut = self.conv1_2(shortcut)
        shortcut = shortcut.replace_feature(self.act1_2(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_2(shortcut.features))

        resA = self.conv2(x)
        resA = resA.replace_feature(self.act2(resA.features))
        resA = resA.replace_feature(self.bn1(resA.features))

        resA = self.conv3(resA)
        resA = resA.replace_feature(self.act3(resA.features))
        resA = resA.replace_feature(self.bn2(resA.features))
        resA = resA.replace_feature(resA.features + shortcut.features)

        return resA


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3, 3), stride=1,
                 pooling=True, drop_out=True, height_pooling=False, indice_key=None, fusion=False):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out

        self.conv1 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef1")
        self.act1 = nn.LeakyReLU(inplace=True)
        self.bn0 = nn.BatchNorm1d(out_filters)

        self.conv1_2 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef2")
        # self.conv1_2 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act1_2 = nn.LeakyReLU(inplace=True)
        self.bn0_2 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef3")
        self.act2 = nn.LeakyReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef4")
        # self.conv3 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act3 = nn.LeakyReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(out_filters)

        if pooling:
            if height_pooling:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=2,
                                                padding=1, indice_key=indice_key, bias=False)
            else:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=(2, 2, 1),
                                                padding=1, indice_key=indice_key, bias=False)
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, img=None):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))

        shortcut = self.conv1_2(shortcut)
        shortcut = shortcut.replace_feature(self.act1_2(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_2(shortcut.features))

        resA = self.conv2(x)
        resA = resA.replace_feature(self.act2(resA.features))
        resA = resA.replace_feature(self.bn1(resA.features))

        resA = self.conv3(resA)
        resA = resA.replace_feature(self.act3(resA.features))
        resA = resA.replace_feature(self.bn2(resA.features))

        resA = resA.replace_feature(resA.features + shortcut.features)

        if self.pooling:
            resB = self.pool(resA)
            return resB, resA
        else:
            return resA


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), indice_key=None, up_key=None):
        super(UpBlock, self).__init__()
        # self.drop_out = drop_out
        self.trans_dilao = conv3x3(in_filters, out_filters, indice_key=indice_key + "new_up")
        self.trans_act = nn.LeakyReLU(inplace=True)
        self.trans_bn = nn.BatchNorm1d(out_filters)

        self.conv1 = conv1x3(out_filters, out_filters, indice_key=indice_key + 'up1')
        self.act1 = nn.LeakyReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv3x1(out_filters, out_filters, indice_key=indice_key + 'up2')
        self.act2 = nn.LeakyReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x3(out_filters, out_filters, indice_key=indice_key + 'up3')
        self.act3 = nn.LeakyReLU(inplace=True)
        self.bn3 = nn.BatchNorm1d(out_filters)
        # self.dropout3 = nn.Dropout3d(p=dropout_rate)

        self.up_subm = spconv.SparseInverseConv3d(out_filters, out_filters, kernel_size=3, indice_key=up_key,
                                                  bias=False)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, skip):
        x = x.replace_feature(x.features.float())

        upA = self.trans_dilao(x)

        # 经过trans_dilao后,值溢出了  inf和-inf都有
        upA = upA.replace_feature(self.trans_act(upA.features))
        # print("------Input------")

        upA = upA.replace_feature(self.trans_bn(upA.features))
        # print("------Output------")

        ## upsample
        upA = self.up_subm(upA)
        upA = upA.replace_feature(upA.features + skip.features)

        upE = self.conv1(upA)
        upE = upE.replace_feature(self.act1(upE.features))
        upE = upE.replace_feature(self.bn1(upE.features))
        # print("nan of upE_conv1", torch.any(torch.isnan(upE.features)))
        upE = self.conv2(upE)
        upE = upE.replace_feature(self.act2(upE.features))
        upE = upE.replace_feature(self.bn2(upE.features))
        # print("nan of upE_conv2", torch.any(torch.isnan(upE.features)))
        upE = self.conv3(upE)
        upE = upE.replace_feature(self.act3(upE.features))
        upE = upE.replace_feature(self.bn3(upE.features))
        # print("nan of upE_conv3", torch.any(torch.isnan(upE.features)))
        return upE


class ReconBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ReconBlock, self).__init__()
        self.conv1 = conv3x1x1(in_filters, out_filters, indice_key=indice_key + "bef1")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.Sigmoid()

        self.conv1_2 = conv1x3x1(in_filters, out_filters, indice_key=indice_key + "bef2")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.Sigmoid()

        self.conv1_3 = conv1x1x3(in_filters, out_filters, indice_key=indice_key + "bef3")
        self.bn0_3 = nn.BatchNorm1d(out_filters)
        self.act1_3 = nn.Sigmoid()

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))

        shortcut2 = self.conv1_2(x)
        shortcut2 = shortcut2.replace_feature(self.bn0_2(shortcut2.features))
        shortcut2 = shortcut2.replace_feature(
            self.act1_2(shortcut2.features))  # 使用amp时，一些非常小的数值会被直接舍入到0，所以需要把tensor的half类型转换到float32

        shortcut3 = self.conv1_3(x)
        shortcut3 = shortcut.replace_feature(self.bn0_3(shortcut3.features))
        shortcut3 = shortcut3.replace_feature(self.act1_3(shortcut3.features))
        shortcut = shortcut.replace_feature(shortcut.features + shortcut2.features + shortcut3.features)

        shortcut = shortcut.replace_feature(shortcut.features * x.features)

        return shortcut


class Asymm_3d_spconv(nn.Module):
    def __init__(self,
                 cfgs,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 nclasses=20, n_height=32, strict=False, init_size=16, dense=True):
        super(Asymm_3d_spconv, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        self.pure_version = cfgs['pure'] if 'pure' in cfgs else False
        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        # print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.dense = dense
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
            4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1, bias=True)

        if self.pure_version:
            self.pool3d = spconv.SparseMaxPool3d((1, 1, 32), indice_key="pool3d")
        else:
            self.upBlock3_ins_heatmap = UpBlock(4 * init_size, 2 * init_size, indice_key="up3_ins_heatmap",
                                                up_key="down2")
            self.upBlock3_ins_offset = UpBlock(4 * init_size, 2 * init_size, indice_key="up3_ins_offset",
                                               up_key="down2")
            self.ReconNet_ins_heatmap = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon_ins_heatmap")
            self.ReconNet_ins_offset = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon_ins_offset")
            self.compress_offset = spconv.SubMConv3d(4 * init_size, 32, indice_key="compress_heatmap", kernel_size=3,
                                                     stride=1, padding=1,
                                                     bias=True)
            self.pool3d_heatmap = spconv.SparseMaxPool3d((1, 1, 32), indice_key="pool3d_heatmap")
            self.pool3d_offset = spconv.SparseMaxPool3d((1, 1, 32), indice_key="pool3d_offset")
            self.logits_offset = nn.Conv2d(32, 2, 3, padding=(1, 0))

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)  # ret.spatial_shape [480, 360, 32]
        # print("\nnan of ret", torch.any(torch.isnan(ret.features)))
        ret = self.downCntx(ret)  # [B, 32,480, 360, 32]
        down1c, down1b = self.resBlock2(ret)  # down1c [B, 64,  240, 180, 16]  ; down1b [B, 64,  480, 360, 32]
        down2c, down2b = self.resBlock3(down1c)  # down2c [B, 128, 120,  90,  8]  ; down2b [B, 128, 240, 180, 16]
        down3c, down3b = self.resBlock4(down2c)  # down3c [B, 256,  60,  45,  8]  ; down3b [B, 256, 120,  90, 8]
        down4c, down4b = self.resBlock5(down3c)  # down4c [B, 512,  30,  23,  8]  ; down4b [B, 512,  60,  45, 8]

        up4e = self.upBlock0(down4c, down4b)  # [B, 512,  60,  45,  8]
        up3e = self.upBlock1(up4e, down3b)  # [B, 256, 120,  90,  8]

        up2e = self.upBlock2(up3e, down2b)  # [B, 128, 240, 180, 16]

        up1e = self.upBlock3(up2e, down1b)  # [B,  64, 480, 360, 32]

        up0e = self.ReconNet(up1e)  # [B,  64, 480, 360, 32]
        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))  # [B, 128, 480, 360, 32]

        logits = self.logits(up0e)  # [1, 20, 480, 360, 32]

        if self.dense:
            logits = logits.dense()
        ######################################################################################
        if self.pure_version:
            bev_features = self.pool3d(up0e)
            return logits, bev_features
        else:
            up1e_ins_heatmap = self.upBlock3_ins_heatmap(up2e, down1b)  # [1, 64, 480, 360, 32]
            up0e_ins_heatmap = self.ReconNet_ins_heatmap(up1e_ins_heatmap)  # [1, 64, 480, 360, 32]
            up0e_ins_heatmap = up0e_ins_heatmap.replace_feature(
                torch.cat((up0e_ins_heatmap.features, up1e_ins_heatmap.features), 1))  # [1, 128, 480, 360, 32]
            up0e_ins_heatmap = self.pool3d_heatmap(up0e_ins_heatmap)  # [1, 128, 480, 360, 1]
            heatmap = up0e_ins_heatmap.dense().squeeze(-1)  # [1, 128, 480, 360]

            up1e_ins_offset = self.upBlock3_ins_offset(up2e, down1b)  # [1, 64, 480, 360, 32]
            up0e_ins_offset = self.ReconNet_ins_offset(up1e_ins_offset)  # [1, 64, 480, 360, 32]
            up0e_ins_offset = up0e_ins_offset.replace_feature(
                torch.cat((up0e_ins_offset.features, up1e_ins_offset.features), 1))  # [1, 128, 480, 360, 32]
            up0e_ins_offset = self.pool3d_offset(up0e_ins_offset)  # [1, 128, 480, 360, 1]
            up0e_ins_offset = self.compress_offset(up0e_ins_offset)  # [1, 32, 480, 360, 1]
            offset = up0e_ins_offset.dense().squeeze(-1)  # [1, 32, 480, 360]
            offset = F.pad(offset, (1, 1, 0, 0), mode='circular')  # [1, 32, 480, 362]
            offset = self.logits_offset(offset)  # [1, 2, 480, 360]

            return logits, heatmap, offset


class BEV_Unet(nn.Module):

    def __init__(self, input_dim, output_dim=1, dilation=1, group_conv=False, input_batch_norm=True, dropout=0.5,
                 circular_padding=True, dropblock=True, pure_version=False):
        super(BEV_Unet, self).__init__()

        self.pure_version = pure_version

        if self.pure_version:
            self.network = PureUNet(input_dim, output_dim, dilation, group_conv, input_batch_norm, dropout,
                                    circular_padding, dropblock)
        else:
            self.network = UNet(input_dim, output_dim, dilation, group_conv, input_batch_norm, dropout,
                                circular_padding, dropblock)

    def forward(self, x):
        return self.network(x)


class UNet(nn.Module):
    def __init__(self, input_dim, output_dim, dilation, group_conv, input_batch_norm, dropout, circular_padding,
                 dropblock):
        super(UNet, self).__init__()
        # encoder
        self.inc = inconv(input_dim, 128, dilation, input_batch_norm, circular_padding)
        self.down1 = down(128, 128, dilation, group_conv, circular_padding)
        self.down2 = down(128, 256, dilation, group_conv, circular_padding)
        self.down3 = down(256, 512, dilation, group_conv, circular_padding)

        # semantic decoder
        self.up2 = up(768, 256, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout)
        self.up3 = up(384, 128, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout)
        self.dropout = nn.Dropout(p=0. if dropblock else dropout)

        self.i_up4_center = up(256, 32, circular_padding, group_conv=group_conv, use_dropblock=dropblock,
                               drop_p=dropout)
        # instance head
        self.i_outc_center = outconv(32, output_dim)

    def forward(self, x):
        # Overview:
        # Inputs:
        #   x: [1,128,480,360]
        # Outputs:
        #   i_x_center: [1, 1, 480, 360]
        x1 = self.inc(x)  # x [1,128,480,360] -> x1 [1, 128, 480, 360]
        x2 = self.down1(x1)  # x1 [1, 128, 480, 360] -> x2 [1, 128, 240, 180]
        x3 = self.down2(x2)  # x2 [1, 128, 240, 180] -> x3 [1, 256, 120, 90]
        x4 = self.down3(x3)  # x3 [1, 256, 120, 90]  -> x4 [1, 512, 60, 45]
        # semantic
        x = self.up2(x4, x3)  # x4 [1, 512, 60, 45] & x3 [1, 256, 120, 90]  -> x [1, 256, 120, 90]
        x = self.up3(x, x2)  # x [1, 256, 120, 90] & x2 [1, 128, 240, 180] -> x [1, 128, 240, 180]

        # x [1, 128, 240, 180] & x1 [1, 128, 480, 360] -> i_x_center [1, 32, 480, 360]
        i_x_center = self.i_up4_center(x, x1)
        # i_x_center [1, 32, 480, 360] -> [1, 1, 480, 360]
        i_x_center = self.i_outc_center(self.dropout(i_x_center))

        return i_x_center


class PureUNet(nn.Module):
    def __init__(self, input_dim, output_dim, dilation, group_conv, input_batch_norm, dropout, circular_padding,
                 dropblock):
        super(PureUNet, self).__init__()
        # encoder
        self.inc = inconv(input_dim, 128, dilation, input_batch_norm, circular_padding)
        self.down1 = down(128, 128, dilation, group_conv, circular_padding)
        self.down2 = down(128, 256, dilation, group_conv, circular_padding)
        self.down3 = down(256, 512, dilation, group_conv, circular_padding)
        self.down4 = down(512, 512, dilation, group_conv, circular_padding)
        # semantic decoder
        self.up1 = up(1024, 256, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout)
        self.up2 = up(512, 128, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout)
        self.up3 = up(256, 64, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout)
        # self.up4 = up(128, 64, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout)
        self.dropout = nn.Dropout(p=0. if dropblock else dropout)
        #######################
        self.i_up4_center = up(192, 32, circular_padding, group_conv=group_conv, use_dropblock=dropblock,
                               drop_p=dropout)
        self.i_up4_offset = up(192, 32, circular_padding, group_conv=group_conv, use_dropblock=dropblock,
                               drop_p=dropout)
        # instance head
        self.i_outc_center = outconv(32, 1)
        self.i_outc_offset = outconv(32, 2)

    def forward(self, x):
        # Overview:
        # Inputs:
        #   x: [1,128,480,360]
        # Outputs:
        #   i_x_center: [1, 1, 480, 360]
        x1 = self.inc(x)  # x [1,128,480,360] -> x1 [1, 128, 480, 360]
        x2 = self.down1(x1)  # x1 [1, 128, 480, 360] -> x2 [1, 128, 240, 180]
        x3 = self.down2(x2)  # x2 [1, 128, 240, 180] -> x3 [1, 256, 120, 90]
        x4 = self.down3(x3)  # x3 [1, 256, 120, 90]  -> x4 [1, 512, 60, 45]
        x5 = self.down4(x4)
        # semantic
        x = self.up1(x5, x4)
        x = self.up2(x, x3)  # x4 [1, 512, 60, 45] & x3 [1, 256, 120, 90]  -> x [1, 256, 120, 90]
        x = self.up3(x, x2)  # x [1, 256, 120, 90] & x2 [1, 128, 240, 180] -> x [1, 128, 240, 180]

        # i_x = self.i_up4(i_x, x1)
        i_x_center = self.i_up4_center(x, x1)  # 128+64
        i_x_center = self.i_outc_center(self.dropout(i_x_center))

        i_x_offset = self.i_up4_offset(x, x1)
        i_x_offset = self.i_outc_offset(self.dropout(i_x_offset))

        return i_x_center, i_x_offset


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, group_conv, dilation=1):
        super(double_conv, self).__init__()
        if group_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, groups=min(out_ch, in_ch)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class double_conv_circular(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, group_conv, dilation=1):
        super(double_conv_circular, self).__init__()
        if group_conv:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=(1, 0), groups=min(out_ch, in_ch)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1, 0), groups=out_ch),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=(1, 0)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1, 0)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        # add circular padding
        x = F.pad(x, (1, 1, 0, 0), mode='circular')
        x = self.conv1(x)
        x = F.pad(x, (1, 1, 0, 0), mode='circular')
        x = self.conv2(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, input_batch_norm, circular_padding):
        super(inconv, self).__init__()
        if input_batch_norm:
            if circular_padding:
                self.conv = nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    double_conv_circular(in_ch, out_ch, group_conv=False, dilation=dilation)
                )
            else:
                self.conv = nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    double_conv(in_ch, out_ch, group_conv=False, dilation=dilation)
                )
        else:
            if circular_padding:
                self.conv = double_conv_circular(in_ch, out_ch, group_conv=False, dilation=dilation)
            else:
                self.conv = double_conv(in_ch, out_ch, group_conv=False, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, group_conv, circular_padding):
        super(down, self).__init__()
        if circular_padding:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv_circular(in_ch, out_ch, group_conv=group_conv, dilation=dilation)
            )
        else:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_ch, out_ch, group_conv=group_conv, dilation=dilation)
            )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, circular_padding, bilinear=True, group_conv=False, use_dropblock=False,
                 drop_p=0.5):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif group_conv:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2, groups=in_ch // 2)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        if circular_padding:
            self.conv = double_conv_circular(in_ch, out_ch, group_conv=group_conv)
        else:
            self.conv = double_conv(in_ch, out_ch, group_conv=group_conv)

        self.use_dropblock = use_dropblock
        if self.use_dropblock:
            self.dropblock = DropBlock2D(block_size=7, drop_prob=drop_p)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        if self.use_dropblock:
            x = self.dropblock(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
