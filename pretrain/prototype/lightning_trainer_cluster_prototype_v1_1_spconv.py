import os
import re

import pytorch_lightning
import numpy as np

if pytorch_lightning.__version__.startswith("1"):
    pl_v = 1
    import pytorch_lightning as pl
    from pytorch_lightning.utilities import rank_zero_only
elif pytorch_lightning.__version__.startswith("2"):
    pl_v = 2
    import lightning.pytorch as pl
    from lightning.pytorch.utilities import rank_zero_only
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pretrain.criterion import NCELoss
from pretrain.prototype.slotcon import DINOHead
from pretrain.prototype.sp_process import convert_spid
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(
        torch.t(Id) * wd)
    return ans


def interpolate_from_bev_features(keypoints, bev_features, batch_size, bev_stride):
    """
    Args:
        keypoints: (N1 + N2 + ..., 4)
        bev_features: (B, C, H, W)
        batch_size:
        bev_stride:

    Returns:
        point_bev_features: (N1 + N2 + ..., C)
    """
    # voxel_size = [0.05, 0.05, 0.1]  # KITTI
    voxel_size = [0.1, 0.1, 0.2]  # nuScenes
    # point_cloud_range = np.array([0., -40., -3., 70.4, 40., 1.], dtype=np.float32)  # KITTI
    point_cloud_range = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], dtype=np.float32)  # nuScenes
    x_idxs = (keypoints[:, 1] - point_cloud_range[0]) / voxel_size[0]
    y_idxs = (keypoints[:, 2] - point_cloud_range[1]) / voxel_size[1]

    x_idxs = x_idxs / bev_stride
    y_idxs = y_idxs / bev_stride

    point_bev_features_list = []
    for k in range(batch_size):
        bs_mask = (keypoints[:, 0] == k)

        cur_x_idxs = x_idxs[bs_mask]
        cur_y_idxs = y_idxs[bs_mask]
        cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
        point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
        point_bev_features_list.append(point_bev_features)

    point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (N1 + N2 + ..., C)
    return point_bev_features


class LightningPretrainSpconv(pl.LightningModule):
    def __init__(self, model_points, model_images, config):
        super().__init__()
        self.model_points = model_points
        self.model_images = model_images
        self._config = config
        self.losses = config["losses"]
        self.train_losses = []
        self.val_losses = []
        self.num_matches = config["num_matches"]
        self.batch_size = config["batch_size"]
        self.num_epochs = config["num_epochs"]
        self.superpixel_size = config["superpixel_size"]

        ####
        self.epoch = 0
        if config["resume_path"] is not None:
            self.epoch = int(
                re.search(r"(?<=epoch=)[0-9]+", config["resume_path"])[0]
            )
        self.criterion = NCELoss(temperature=config["NCE_temperature"])
        self.cluster_criterion = ClusterLoss(temperature=1.0)
        self.working_dir = os.path.join(config["working_dir"], config["datetime"])
        if os.environ.get("LOCAL_RANK", 0) == 0:
            os.makedirs(self.working_dir, exist_ok=True)

        self.without_compression = config['without_compression'] if 'without_compression' in config else False
        ##### prototype learning #####
        self.point_prototype_temperature = config['temperature']
        self.dim_out = config['dim_out']
        self.dim_hidden = config['dim_hidden']
        self.predictor_slot = DINOHead(self.dim_out * 2, hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out)
        ###################################
        self.num_pixel_prototype = config['num_pixel_prototype']  # pixel的prototype需要更新
        self.num_point_prototype = config['num_point_prototype']
        self.prototype_dim = config['dim_prototype']
        # 初始化2d/3d prototype
        self.register_buffer("pixel_prototype", torch.zeros(self.num_pixel_prototype, self.prototype_dim))
        self.register_buffer("point_prototype", torch.zeros(self.num_pixel_prototype, self.prototype_dim))
        ###################################

        self.superpixels_type = config["superpixels_type"]
        self.cluster_uses_all_superpixel = config["cluster_uses_all_superpixel"]
        self.cluster_interval = config['cluster_interval']
        ###################################
        self.all_superpixel_features = None
        self.paired_superpixel_features = None
        self.paired_superpoint_features = None
        self.all_sp_image_path_list = None
        self.paired_sp_image_path_list = None
        self.all_sp_ids = None
        self.paired_sp_ids = None
        self.use_cluster_loss = False

    def configure_optimizers(self):
        optimizer = optim.SGD(
            list(self.model_points.parameters())
            + list(self.model_images.parameters())
            + list(self.predictor_slot.parameters())
            ,
            lr=self._config["lr"],
            momentum=self._config["sgd_momentum"],
            dampening=self._config["sgd_dampening"],
            weight_decay=self._config["weight_decay"],
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_epochs)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # 相比baseline
        self.model_points.train()
        self.model_images.eval()
        self.model_images.decoder.train()
        #############################################################################################################
        if not self.without_compression:
            # 有高度压缩
            output_points = self.model_points(batch["voxels"], batch["coordinates"])  # (16,64,128,128)
            output_points = interpolate_from_bev_features(
                batch["pc"],
                output_points,
                self.batch_size,
                self.model_points.bev_stride
            )
        else:
            output_points = self.model_points(batch["voxels"], batch["coordinates"]).features
        output_images = self.model_images(batch["input_I"])  # [96,64,224,416]
        #############################################################################################################
        losses = []
        one_hot_P, one_hot_I, sp_sem_id = self.get_superpixel_one_hot_index(batch)
        # ----------------------------
        # loss 1
        superpixel_feats = one_hot_I @ output_images.permute(0, 2, 3, 1).flatten(0, 2)
        superpixel_feats = superpixel_feats / (torch.sparse.sum(input=one_hot_I, dim=1).to_dense()[:, None] + 1e-6)
        pairing_points = batch["pairing_points"]  # 存在匹配关系的point坐标
        pairing_points_feats = output_points[pairing_points]
        superpoint_feats_sum = one_hot_P @ pairing_points_feats  # (N,M) @ (M,C) -> (N,C)
        superpoint_num = torch.sparse.sum(one_hot_P, 1).to_dense()[:, None] + 1e-6  # [3072] 有1530个superpixel是没用的.
        superpoint_feats = superpoint_feats_sum / superpoint_num  # 获取超点的平均特征
        mask = torch.where(superpoint_num != 1e-6)[0]
        valid_superpoint_feats = superpoint_feats[mask, :]
        valid_superpixel_feats = superpixel_feats[mask, :]

        torch.cuda.empty_cache()
        sp_loss = self.criterion(valid_superpoint_feats, valid_superpixel_feats)

        losses.append(sp_loss)
        if self.use_cluster_loss:
            torch.cuda.empty_cache()
            sp_sem_id = sp_sem_id[mask]  # 获得到每个superpoint/superpixel的 sem_id (N)
            sp_sem_id = convert_spid(self.superpixels_type, sp_sem_id)
            mixed_prototype = self.predictor_slot(
                torch.cat(
                    [
                        self.point_prototype,
                        self.pixel_prototype
                    ], dim=1)
            )  # (150,64)
            instance_prototype_loss = self.cluster_criterion(valid_superpoint_feats, mixed_prototype, sp_sem_id)
            losses.append(instance_prototype_loss)
        else:

            instance_prototype_loss = 0.0
        loss = torch.mean(torch.stack(losses))

        if pl_v == 1:
            self.log_dict(
                {
                    "t_ls": loss,
                    "t_sp_ls": sp_loss,
                    "t_cluster_ls": instance_prototype_loss,
                    # "t_match_ls": prototype_match_loss,
                },
                on_step=True, on_epoch=True, prog_bar=True, logger=True
            )
        elif pl_v == 2:
            self.log_dict(
                {
                    "t_ls": loss,
                    "t_sp_ls": sp_loss,
                    "t_cluster_ls": instance_prototype_loss,
                    # "t_match_ls": prototype_match_loss,
                },
                on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
            )
        self.train_losses.append(loss.detach().cpu())
        return loss

    def on_train_epoch_start(self) -> None:
        # 0->1 ,1
        if self.current_epoch >= self.cluster_interval:
            self.use_cluster_loss = True
        print(f"current_epoch:{self.current_epoch}; "
              f"cluster_interval:{self.cluster_interval}; "
              f"use_cluster_loss:{self.use_cluster_loss}")

    def on_train_epoch_end(self):
        self.epoch += 1
        if self.epoch == self.num_epochs:
            self.save()

    # ----------------------------------------------------
    def on_validation_epoch_start(self):
        # 超像素/超点的特征
        self.all_superpixel_features = []
        self.paired_superpixel_features = []
        self.paired_superpoint_features = []
        # 超像素对应图像路径的列表
        self.all_sp_image_path_list = []
        self.paired_sp_image_path_list = []
        # 超像素对应标号的列表
        self.all_sp_ids = []
        self.paired_sp_ids = []

    def validation_step(self, batch, batch_idx):
        ## 验证阶段使用的是训练集
        # 相比baseline
        self.model_points.eval()
        self.model_images.eval()

        if not self.without_compression:
            output_points = self.model_points(batch["voxels"], batch["coordinates"])
            output_points = interpolate_from_bev_features(
                batch["pc"],
                output_points,
                self.batch_size,
                self.model_points.bev_stride)
        else:
            output_points = self.model_points(batch["voxels"], batch["coordinates"]).features
        output_images = self.model_images(batch["input_I"])  # [96,64,224,416]
        #############################################################################################################
        one_hot_P, one_hot_I, sp_id = self.get_superpixel_one_hot_index(batch, return_image_id=False)
        # ----------------------------
        superpixel_feats = one_hot_I @ output_images.permute(0, 2, 3, 1).flatten(0, 2)
        superpixel_feats = superpixel_feats / (torch.sparse.sum(input=one_hot_I, dim=1).to_dense()[:, None] + 1e-6)
        pairing_points = batch["pairing_points"]  # 存在匹配关系的point坐标
        pairing_points_feats = output_points[pairing_points]
        superpoint_feats_sum = one_hot_P @ pairing_points_feats  # (N,M) @ (M,C) -> (N,C)
        superpoint_num = torch.sparse.sum(one_hot_P, 1).to_dense()[:, None] + 1e-6  # [3072] 有1530个superpixel是没用的.
        superpoint_feats = superpoint_feats_sum / superpoint_num  # 获取超点的平均特征
        mask = torch.where(superpoint_num != 1e-6)[0]
        paired_superpoint_feats = superpoint_feats[mask, :]
        paired_superpixel_feats = superpixel_feats[mask, :]
        # --------------------------------------------------------
        sp_id = convert_spid(self.superpixels_type, sp_id)
        paired_sp_id = sp_id[mask]  # 获得到每个superpoint/superpixel的 sem_id (N)
        paired_sp_id = convert_spid(self.superpixels_type, paired_sp_id)
        # -------------------------------------------------------

        self.all_superpixel_features.append(superpixel_feats)  # List[Tensor]
        self.paired_superpixel_features.append(paired_superpixel_feats)  # List[Tensor]
        self.paired_superpoint_features.append(paired_superpoint_feats)  # List[Tensor]

        self.all_sp_ids.append(sp_id)  # List[Tensor]
        self.paired_sp_ids.append(paired_sp_id)  # List[Tensor]

    def on_validation_epoch_end(self):

        all_superpixel_features = torch.cat(self.all_superpixel_features)
        all_sp_ids = torch.cat(self.all_sp_ids)

        paired_superpixel_features = torch.cat(self.paired_superpixel_features)
        paired_superpoint_features = torch.cat(self.paired_superpoint_features)
        paired_sp_ids = torch.cat(self.paired_sp_ids)
        # -----------------------------------------------------------------------
        pixel_prototype = torch.zeros(self.num_pixel_prototype, self.prototype_dim)
        point_prototype = torch.zeros(self.num_point_prototype, self.prototype_dim)

        # pixel的特征可以选择来自2d-3d匹配的区域
        for cluster_idx in tqdm(range(self.num_pixel_prototype)):
            # dino 0 开始
            if self.cluster_uses_all_superpixel:
                indices = all_sp_ids == cluster_idx
            else:
                indices = paired_sp_ids == cluster_idx
            #
            if torch.count_nonzero(indices) > 0:
                superpixel_features = all_superpixel_features if self.cluster_uses_all_superpixel else paired_superpixel_features
                cluster_feature = superpixel_features[indices].mean(0, keepdims=True)
                pixel_prototype[cluster_idx] = cluster_feature

        for cluster_idx in tqdm(range(self.num_point_prototype)):
            indices = paired_sp_ids == cluster_idx
            if torch.count_nonzero(indices) > 0:
                cluster_feature = paired_superpoint_features[indices].mean(0, keepdims=True)
                point_prototype[cluster_idx] = cluster_feature

        # 多进程同步
        pixel_prototype = self.all_gather(pixel_prototype)  # (N_sp,64) -> (N_gpu,N_sp,64)
        point_prototype = self.all_gather(point_prototype)
        pixel_prototype = torch.mean(pixel_prototype, dim=0)  # (N_gpu,N_sp,64) -> (N_sp,64)
        point_prototype = torch.mean(point_prototype, dim=0)  # (N_gpu,N_sp,64) -> (N_sp,64)
        self.pixel_prototype = pixel_prototype.cuda()
        self.point_prototype.data = point_prototype.cuda()
        #
        # if self.current_epoch>=:
        self.use_cluster_loss = True

    def get_superpixel_one_hot_index(self, batch, return_image_id=False):
        # compute a superpoints to superpixels loss using superpixels
        # torch.cuda.empty_cache()  # This method is extremely memory intensive
        num_images = batch['input_I'].shape[0]
        superpixels = batch["superpixels"]
        superpixels_id = superpixels
        superpixel_size = int(superpixels.max()) + 1
        # print(superpixels.max(), int(superpixels.max()) + 1, int(superpixels.max() + 1))
        pairing_images = batch["pairing_images"]
        pairing_points = batch["pairing_points"]

        superpixels = (
                torch.arange(
                    0,
                    num_images * superpixel_size,  # 改成图像数量即可
                    superpixel_size,
                    device=self.device,
                )[:, None, None] + superpixels
        )  #
        m = tuple(pairing_images.cpu().T.long())

        superpixels_I = superpixels.flatten()
        idx_P = torch.arange(pairing_points.shape[0], device=superpixels.device)
        total_pixels = superpixels_I.shape[0]
        idx_I = torch.arange(total_pixels, device=superpixels.device)
        # --------------------------------------------------
        superpixels_id_I = superpixels_id.flatten()
        superpixels_image_id = torch.zeros_like(superpixels_id)  # [48,224,416]
        # --------------------------------------------------
        if return_image_id:
            image_ids = torch.arange(0,  # start
                                     num_images,  # end
                                     1,  # step_size
                                     device=self.device)[:, None, None]  # (48,1,1)
            image_ids = image_ids.expand(list(superpixels_image_id.shape))  # (48,224,416)
            image_ids = image_ids.flatten()
        ###########################################################
        with torch.no_grad():
            # ----------------------------------------------
            one_hot_P = torch.sparse_coo_tensor(
                torch.stack((
                    superpixels[m], idx_P
                ), dim=0),
                torch.ones(pairing_points.shape[0], device=superpixels.device),
                (superpixels.shape[0] * superpixel_size, pairing_points.shape[0])
            )
            # ----------------------------------------------
            one_hot_I = torch.sparse_coo_tensor(
                torch.stack((
                    superpixels_I, idx_I
                ), dim=0),
                torch.ones(total_pixels, device=superpixels.device),
                (superpixels.shape[0] * superpixel_size, total_pixels)
            )
            # ----------------------------------------------
            max_sem_id = torch.max(superpixels_id_I)
            sp_sem_id = torch.sparse_coo_tensor(
                torch.stack((
                    superpixels_I, superpixels_id_I
                ), dim=0),  # indices
                torch.ones(total_pixels, device=superpixels.device),  # values
                (superpixels.shape[0] * superpixel_size, max_sem_id + 1)  # size
            )
            sp_sem_id = sp_sem_id.to_dense()
            sp_sem_id = torch.argmax(sp_sem_id, dim=1)
            # ----------------------------------------------
            if return_image_id:
                sp_image_id = torch.sparse_coo_tensor(
                    torch.stack((
                        superpixels_I, image_ids
                    ), dim=0),  # indices
                    torch.ones(total_pixels, device=superpixels.device),  # values
                    (superpixels.shape[0] * superpixel_size, num_images)  # size
                )
                sp_image_id = sp_image_id.to_dense()  # (7200,48)
                sp_image_id = torch.argmax(sp_image_id, dim=1)  # (7200)
        # ------------------------------------
        if return_image_id:
            return one_hot_P, one_hot_I, sp_sem_id, sp_image_id
        else:
            return one_hot_P, one_hot_I, sp_sem_id

    @rank_zero_only
    def save(self):
        path = os.path.join(self.working_dir, "model.pt")
        torch.save(
            {
                "model_points": self.model_points.state_dict(),
                "model_images": self.model_images.state_dict(),
                "epoch": self.epoch,
                "config": self._config,
            },
            path,
        )


class ClusterLoss(nn.Module):

    def __init__(self, temperature):
        super(ClusterLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, k, q, tareget):
        # k: (N,d)
        # q: (M,d)
        # tareget: (N)
        logits = torch.mm(k, q.transpose(1, 0))  # (N,M)
        out = torch.div(logits, self.temperature)
        out = out.contiguous()  #

        loss = self.criterion(out, tareget)
        return loss
