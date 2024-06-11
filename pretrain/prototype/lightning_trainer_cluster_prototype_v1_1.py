import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import MinkowskiEngine as ME
import pytorch_lightning
from tqdm import tqdm

if pytorch_lightning.__version__.startswith("1"):
    pl_v = 1
    import pytorch_lightning as pl
    from pytorch_lightning.utilities import rank_zero_only
elif pytorch_lightning.__version__.startswith("2"):
    pl_v = 2
    import lightning.pytorch as pl
    from lightning.pytorch.utilities import rank_zero_only
######################
from pretrain.criterion import NCELoss
from pretrain.prototype.slotcon import DINOHead
from pretrain.prototype.sp_process import convert_spid


######################
class LightningPretrain(pl.LightningModule):
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
        nce_temperature = config["NCE_temperature"]
        cluster_temperature = 1.0 if 'cluster_temperature' not in config else config["cluster_temperature"]
        self.criterion = NCELoss(temperature=nce_temperature)
        self.cluster_criterion = ClusterLoss(temperature=cluster_temperature)
        print(f"NCE_temperature:{nce_temperature}; cluster_temperature:{cluster_temperature}")
        self.working_dir = config['working_dir']
        if os.environ.get("LOCAL_RANK", 0) == 0:
            os.makedirs(self.working_dir, exist_ok=True)
        ##### prototype learning #####
        self.uni_modality_generation_op = config.get("uni_modality_generation_op", 'avg')  #
        assert self.uni_modality_generation_op in ['avg', 'max', 'sum']

        self.point_prototype_temperature = config['temperature']
        self.dim_out = config['dim_out']
        self.dim_hidden = config['dim_hidden']
        self.num_pixel_prototype = config['num_pixel_prototype']  # pixel的prototype需要更新
        self.num_point_prototype = config['num_point_prototype']
        self.prototype_dim = config['dim_prototype']
        ##### 初始化2d/3d prototype #####
        self.register_buffer("pixel_prototype", torch.zeros(self.num_pixel_prototype, self.prototype_dim))
        self.register_buffer("point_prototype", torch.zeros(self.num_pixel_prototype, self.prototype_dim))
        ###################################
        self.predictor_slot = DINOHead(self.dim_out * 2, hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out)

        self.superpixels_type = config["superpixels_type"]
        self.cluster_uses_all_superpixel = config["cluster_uses_all_superpixel"]
        self.cluster_interval = config['cluster_interval']
        ###################################
        # 局部变量
        self.all_superpixel_features = None
        self.paired_superpixel_features = None
        self.paired_superpoint_features = None
        self.all_sp_image_path_list = None
        self.paired_sp_image_path_list = None
        self.all_sp_ids = None
        self.paired_sp_ids = None
        self.use_cluster_loss = False
        ###################################
        # 全局变量
        self.global_paired_sp_ids = None

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

        sparse_input = ME.SparseTensor(batch["sinput_F"], batch["sinput_C"])  # 特征 & 坐标
        output_points = self.model_points(sparse_input)  # (voxel)
        output_images = self.model_images(batch["input_I"])  # [96,64,224,416]
        #############################################################################################################
        losses = []
        output_points = output_points.F
        one_hot_P, one_hot_I, sp_sem_id, sp_image_id \
            = self.get_superpixel_one_hot_index(batch, return_image_id=True)
        cur_device = output_points.device
        batch_image_path = batch['image_path']
        image_list = []
        for i in batch_image_path:
            image_list += i
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
        # torch.cuda.empty_cache()
        sp_loss = self.criterion(valid_superpoint_feats, valid_superpixel_feats)
        losses.append(sp_loss)
        if self.use_cluster_loss:
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
                },
                on_step=True, on_epoch=True, prog_bar=True, logger=True
            )
        elif pl_v == 2:
            self.log_dict(
                {
                    "t_ls": loss,
                    "t_sp_ls": sp_loss,
                    "t_cluster_ls": instance_prototype_loss,
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
        if self.epoch >= self.num_epochs - 1:
            self.save(save_weights=True, final=True)

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

        if self.use_cluster_loss:
            self.save(save_weights=False, final=False)

    def validation_step(self, batch, batch_idx):
        ## 验证阶段使用的是训练集
        # 相比baseline
        self.model_points.eval()
        self.model_images.eval()

        sparse_input = ME.SparseTensor(batch["sinput_F"], batch["sinput_C"])  # 特征 & 坐标
        output_points = self.model_points(sparse_input).F  # (voxel)
        output_images = self.model_images(batch["input_I"])  # [96,64,224,416]
        #############################################################################################################
        one_hot_P, one_hot_I, sp_id, sp_image_id = self.get_superpixel_one_hot_index(batch, return_image_id=True)
        cur_device = output_points.device
        batch_image_path = batch['image_path']
        image_list = []
        for i in batch_image_path:
            image_list += i
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

        paired_sp_image_id = sp_image_id[mask].detach().cpu().tolist()
        sp_image_path_list = []
        paired_sp_image_path_list = []
        for image_id in sp_image_id:
            sp_image_path_list.append(image_list[image_id])
        for image_id in paired_sp_image_id:
            paired_sp_image_path_list.append(image_list[image_id])
        # -------------------------------------------------------
        self.all_superpixel_features.append(superpixel_feats)  # List[Tensor]
        self.paired_superpixel_features.append(paired_superpixel_feats)  # List[Tensor]
        self.paired_superpoint_features.append(paired_superpoint_feats)  # List[Tensor]

        self.all_sp_image_path_list += sp_image_path_list  # List[str]
        self.paired_sp_image_path_list += paired_sp_image_path_list  # List[str]

        self.all_sp_ids.append(sp_id)  # List[Tensor]
        self.paired_sp_ids.append(paired_sp_id)  # List[Tensor]

    def on_validation_epoch_end(self):
        paired_superpixel_features = torch.cat(self.paired_superpixel_features)
        paired_superpoint_features = torch.cat(self.paired_superpoint_features)
        paired_sp_ids = torch.cat(self.paired_sp_ids)
        self.global_paired_sp_ids = paired_sp_ids
        # -----------------------------------------------------------------------
        pixel_prototype = torch.zeros(self.num_pixel_prototype, self.prototype_dim)
        point_prototype = torch.zeros(self.num_point_prototype, self.prototype_dim)

        # pixel的特征可以选择来自2d-3d匹配的区域
        for cluster_idx in tqdm(range(self.num_pixel_prototype)):
            indices = paired_sp_ids == cluster_idx
            if torch.count_nonzero(indices) > 0:
                superpixel_features = paired_superpixel_features
                #
                if self.uni_modality_generation_op == 'avg':
                    cluster_feature = superpixel_features[indices].mean(0, keepdims=True)
                elif self.uni_modality_generation_op == 'max':
                    cluster_feature, _ = superpixel_features[indices].max(0, keepdims=True)
                elif self.uni_modality_generation_op == 'sum':
                    cluster_feature = superpixel_features[indices].sum(0, keepdims=True)
                #
                pixel_prototype[cluster_idx] = cluster_feature

        for cluster_idx in tqdm(range(self.num_point_prototype)):
            indices = paired_sp_ids == cluster_idx
            if torch.count_nonzero(indices) > 0:
                #
                if self.uni_modality_generation_op == 'avg':
                    cluster_feature = paired_superpoint_features[indices].mean(0, keepdims=True)
                elif self.uni_modality_generation_op == 'max':
                    cluster_feature, _ = paired_superpoint_features[indices].max(0, keepdims=True)
                elif self.uni_modality_generation_op == 'sum':
                    cluster_feature = paired_superpoint_features[indices].sum(0, keepdims=True)
                #
                # cluster_feature = paired_superpoint_features[indices].mean(0, keepdims=True)
                point_prototype[cluster_idx] = cluster_feature

        # 多进程同步
        pixel_prototype = self.all_gather(pixel_prototype)  # (N_sp,64) -> (N_gpu,N_sp,64)
        point_prototype = self.all_gather(point_prototype)
        pixel_prototype = torch.mean(pixel_prototype, dim=0)  # (N_gpu,N_sp,64) -> (N_sp,64)
        point_prototype = torch.mean(point_prototype, dim=0)  # (N_gpu,N_sp,64) -> (N_sp,64)
        self.pixel_prototype = pixel_prototype.cuda()
        self.point_prototype = point_prototype.cuda()
        #
        self.use_cluster_loss = True
        self.save(save_weights=False, final=False)

    def get_superpixel_one_hot_index(self, batch, return_image_id=False, device=None):
        cur_device = device if device is not None else self.device
        ##################################################################
        ## compute a superpoints to superpixels loss using superpixels  ##
        ##################################################################
        num_images = batch['input_I'].shape[0]
        superpixels = batch["superpixels"]
        superpixels_id = superpixels
        superpixel_size = int(superpixels.max()) + 1
        pairing_images = batch["pairing_images"]
        pairing_points = batch["pairing_points"]

        superpixels = (
                torch.arange(
                    0,
                    num_images * superpixel_size,  # 改成图像数量即可
                    superpixel_size,
                    device=cur_device,
                )[:, None, None] + superpixels
        )  #
        m = tuple(pairing_images.cpu().T.long())

        superpixels_I = superpixels.flatten()
        idx_P = torch.arange(pairing_points.shape[0], device=cur_device)
        total_pixels = superpixels_I.shape[0]
        idx_I = torch.arange(total_pixels, device=cur_device)
        # --------------------------------------------------
        superpixels_id_I = superpixels_id.flatten()
        superpixels_image_id = torch.zeros_like(superpixels_id)  # [48,224,416]
        # --------------------------------------------------
        if return_image_id:
            image_ids = torch.arange(0,  # start
                                     num_images,  # end
                                     1,  # step_size
                                     device=cur_device)[:, None, None]  # (48,1,1)
            image_ids = image_ids.expand(list(superpixels_image_id.shape))  # (48,224,416)
            image_ids = image_ids.flatten()
        ###########################################################
        with torch.no_grad():
            # ----------------------------------------------
            one_hot_P = torch.sparse_coo_tensor(
                torch.stack((
                    superpixels[m], idx_P
                ), dim=0),
                torch.ones(pairing_points.shape[0], device=cur_device),
                (superpixels.shape[0] * superpixel_size, pairing_points.shape[0])
            )
            # ----------------------------------------------
            one_hot_I = torch.sparse_coo_tensor(
                torch.stack((
                    superpixels_I, idx_I
                ), dim=0),
                torch.ones(total_pixels, device=cur_device),
                (superpixels.shape[0] * superpixel_size, total_pixels)
            )
            # ----------------------------------------------
            max_sem_id = torch.max(superpixels_id_I)
            sp_sem_id = torch.sparse_coo_tensor(
                torch.stack((
                    superpixels_I, superpixels_id_I
                ), dim=0),  # indices
                torch.ones(total_pixels, device=cur_device),  # values
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
                    torch.ones(total_pixels, device=cur_device),  # values
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
    def save(self, save_weights=True, final=True):
        if final:
            weight_path = os.path.join(self.working_dir, "final_model_cp_v1_1.pt")
            prototype_path = os.path.join(self.working_dir, "final_prototype_cp_v1_1.pt")
        else:
            weight_path = os.path.join(self.working_dir, f"model_cp_v1_1_{self.current_epoch}.pt")
            prototype_path = os.path.join(self.working_dir, f"prototype_cp_v1_1_{self.current_epoch}.pt")

        if save_weights:
            # 保存2d/3d model的权重，以及原型相关的权重
            torch.save(
                {
                    "model_points": self.model_points.state_dict(),
                    "model_images": self.model_images.state_dict(),
                    "predictor_slot": self.predictor_slot.state_dict(),
                    "pixel_prototype": self.pixel_prototype,
                    "point_prototype": self.point_prototype,
                    "global_paired_sp_ids": self.global_paired_sp_ids,
                    "epoch": self.current_epoch,
                    "config": self._config,
                }, weight_path,
            )
        else:
            # 仅保存原型相关的权重
            torch.save(
                {
                    "predictor_slot": self.predictor_slot.state_dict(),
                    "pixel_prototype": self.pixel_prototype,
                    "point_prototype": self.point_prototype,
                    "global_paired_sp_ids": self.global_paired_sp_ids,
                    "epoch": self.current_epoch,
                    "config": self._config,
                }, prototype_path
            )

    ####################################
    def switch_model(self):
        if self.model_points.training:
            self.model_points.eval()
            self.model_images.eval()
        else:
            self.model_points.train()
            self.model_images.train()

    ####################################
    def inference(self, batch):
        batch = self.batch2cuda(batch)
        sparse_input = ME.SparseTensor(batch["sinput_F"], batch["sinput_C"])  # 特征 & 坐标
        input_images = batch["input_I"]
        output_voxels = self.model_points(sparse_input).F  # [N,64]
        output_images = self.model_images(input_images)  # [96,64,224,416]

        pred_voxels = []
        pred_points = []  # point-wise
        gt_points = []
        offset = 0
        for i, lb in enumerate(batch["len_batch"]):
            cur_voxels_mask = batch["sinput_C"][:, 0] == i
            cur_voxels = output_voxels[cur_voxels_mask]
            pred_voxels.append(cur_voxels)
            inverse_indexes = batch["inverse_indexes"][i]
            predictions = output_voxels[inverse_indexes + offset]
            pred_points.append(predictions)
            offset += lb
        # --------------------------------------------
        # 单独存储每个image的superpixel
        device = input_images.device
        one_hot_P, one_hot_I, sp_sem_id, sp_image_id \
            = self.get_superpixel_one_hot_index(batch, return_image_id=True, device=device)

        pairing_points = batch["pairing_points"].long()  # 存在匹配关系的point坐标
        pairing_points_feats = output_voxels[pairing_points]
        superpoint_feat_list = []
        superpixel_feat_list = []
        unique_image_ids = len(torch.unique(sp_image_id))  # 统计出image的总数
        image_list = []
        for image_id in range(unique_image_ids):
            # sp_sem_id中sem_id==0的sp为无效sp。
            cur_sp_mask = sp_image_id == image_id
            cur_one_hot_I = one_hot_I * cur_sp_mask.reshape(-1, 1)
            cur_one_hot_P = one_hot_P * cur_sp_mask.reshape(-1, 1)
            cur_superpixel_feats = cur_one_hot_I @ output_images.permute(0, 2, 3, 1).flatten(0, 2)
            cur_superpixel_feats = \
                cur_superpixel_feats / (torch.sparse.sum(input=cur_one_hot_I, dim=1).to_dense()[:, None] + 1e-6)
            cur_superpoint_feats_sum = cur_one_hot_P @ pairing_points_feats
            cur_superpoint_num = torch.sparse.sum(one_hot_P, 1).to_dense()[:, None] + 1e-6
            cur_superpoint_feats = cur_superpoint_feats_sum / cur_superpoint_num
            cur_mask = torch.where(cur_superpoint_num != 1e-6)[0]
            cur_valid_superpoint_feats = cur_superpoint_feats[cur_mask, :]
            cur_valid_superpixel_feats = cur_superpixel_feats[cur_mask, :]
            superpoint_feat_list.append(cur_valid_superpoint_feats)
            superpixel_feat_list.append(cur_valid_superpixel_feats)
            image_list.append(batch['image_path'][image_id // 6][image_id % 6])

        output_dict = {
            "pred_voxels": pred_voxels,
            "output_images": output_images,
            "pred_points": pred_points,
            "valid_superpoint_feats": superpoint_feat_list,
            "valid_superpixel_feats": superpixel_feat_list,
            "image_list": image_list,
        }

        return output_dict

    ####################################
    def batch2cuda(self, batch):
        batch['sinput_F'] = batch['sinput_F'].cuda()
        batch['sinput_C'] = batch['sinput_C'].cuda()
        batch['input_I'] = batch['input_I'].cuda()
        batch['pairing_points'] = batch['pairing_points'].cuda()
        batch['pairing_images'] = batch['pairing_images'].cuda()
        batch['superpixels'] = batch['superpixels'].cuda()
        batch['inverse_indexes'] = [x.cuda() for x in batch['inverse_indexes']]

        return batch

    def get_mixed_prototype(self):
        mixed_prototype = self.predictor_slot(
            torch.cat(
                [
                    self.point_prototype,
                    self.pixel_prototype
                ], dim=1)
        )  # (150,64)
        return mixed_prototype


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
