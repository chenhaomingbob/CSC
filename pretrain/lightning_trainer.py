import os
import re
import torch
import numpy as np
import torch.optim as optim
import MinkowskiEngine as ME
import pytorch_lightning
import torch.nn.functional as F

if pytorch_lightning.__version__.startswith("1"):
    import pytorch_lightning as pl
    from pytorch_lightning.utilities import rank_zero_only
elif pytorch_lightning.__version__.startswith("2"):
    import lightning.pytorch as pl
    from lightning.pytorch.utilities import rank_zero_only
from pretrain.criterion import NCELoss


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
        self.epoch = 0
        if config["resume_path"] is not None:
            self.epoch = int(
                re.search(r"(?<=epoch=)[0-9]+", config["resume_path"])[0]
            )
        self.criterion = NCELoss(temperature=config["NCE_temperature"])
        self.working_dir = os.path.join(config["working_dir"], config["datetime"])
        if os.environ.get("LOCAL_RANK", 0) == 0:
            os.makedirs(self.working_dir, exist_ok=True)

    def configure_optimizers(self):
        optimizer = optim.SGD(
            list(self.model_points.parameters()) + list(self.model_images.parameters()),
            lr=self._config["lr"],
            momentum=self._config["sgd_momentum"],
            dampening=self._config["sgd_dampening"],
            weight_decay=self._config["weight_decay"],
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_epochs)
        return [optimizer], [scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def training_step(self, batch, batch_idx):
        self.model_points.train()

        sparse_input = ME.SparseTensor(batch["sinput_F"], batch["sinput_C"])
        output_points = self.model_points(sparse_input).F
        # self.model_images.eval()
        # self.model_images.decoder.train()  # decoder是可以train的
        # output_images = self.model_images(batch["input_I"])  # [96.64,224,416]
        torch.cuda.empty_cache()
        if self._config['images_encoder'] == 'maskclip':
            self.model_images.eval()
            output_images = self.model_images(batch["input_I"])
            output_images_feats, output_images_pred = output_images
            output_images = output_images_feats
        else:
            self.model_images.eval()
            self.model_images.decoder.train()
            output_images = self.model_images(batch["input_I"])  # [96,64,224,416]
        del batch["sinput_F"]
        del batch["sinput_C"]
        del sparse_input
        # each loss is applied independtly on each GPU
        losses = [
            getattr(self, loss)(batch, output_points, output_images)
            for loss in self.losses  # loss_superpixels_average
        ]
        loss = torch.mean(torch.stack(losses))

        torch.cuda.empty_cache()
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
        )
        self.train_losses.append(loss.detach().cpu())
        return loss

    def loss(self, batch, output_points, output_images):
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        idx = np.random.choice(pairing_points.shape[0], self.num_matches, replace=False)
        k = output_points[pairing_points[idx]]
        m = tuple(pairing_images[idx].T.long())
        q = output_images.permute(0, 2, 3, 1)[m]
        return self.criterion(k, q)

    def loss_superpixels_average(self, batch, output_points, output_images):
        # compute a superpoints to superpixels loss using superpixels
        torch.cuda.empty_cache()  # This method is extremely memory intensive
        superpixels = batch["superpixels"]  # batch中每个pixel的超像素id  (16*6,224,416)
        pairing_images = batch["pairing_images"]  # 点云点投影到image上对应的pixel坐标.  (211471,3)
        pairing_points = batch["pairing_points"]  # 能投影至图像的点的下标.   (211471,)

        superpixels = (
                torch.arange(
                    start=0,
                    end=output_images.shape[0] * self.superpixel_size,
                    step=self.superpixel_size,  # 150
                    device=self.device,
                )[:, None, None] + superpixels
        )  # [96,224,416]
        m = tuple(pairing_images.cpu().T.long())  # pixel坐标  [3,n]   [batch_id,x,y]

        superpixels_I = superpixels.flatten()
        idx_P = torch.arange(pairing_points.shape[0], device=superpixels.device)
        total_pixels = superpixels_I.shape[0]
        idx_I = torch.arange(total_pixels, device=superpixels.device)

        with torch.no_grad():
            one_hot_P = torch.sparse_coo_tensor(
                indices=torch.stack((
                    superpixels[m], idx_P
                ), dim=0),  # 坐标
                values=torch.ones(pairing_points.shape[0], device=superpixels.device),
                size=(superpixels.shape[0] * self.superpixel_size, pairing_points.shape[0])  #
            )  # 超像素与点的匹配对 (24576,218051)  超像素与那些point匹配. (图像个数*超像素个数, 关联的点)

            one_hot_I = torch.sparse_coo_tensor(
                indices=torch.stack((
                    superpixels_I, idx_I
                ), dim=0),
                values=torch.ones(total_pixels, device=superpixels.device),
                size=(superpixels.shape[0] * self.superpixel_size, total_pixels)
            )  # 超像素与图像的匹配对 (24576,8945664)

        k = one_hot_P @ output_points[pairing_points]  # 超像素的投影点特征的求和/ 仅求outputs_point的
        k = k / (torch.sparse.sum(one_hot_P, 1).to_dense()[:, None] + 1e-6)  # 超像素的平均投影点特征
        q = one_hot_I @ output_images.permute(0, 2, 3, 1).flatten(0, 2)  # 超像素的图像特征的求和
        q = q / (torch.sparse.sum(one_hot_I, 1).to_dense()[:, None] + 1e-6)  # 超像素的平均图像特征

        mask = torch.where(k[:, 0] != 0)  # 有投影点的超像素mask
        k = k[mask]  # 有投影点的超像素的点特征
        q = q[mask]  # 有投影点的超像素的图像特征

        return self.criterion(k, q)

    def loss_superpixels_distill(self, batch, output_points, output_images):
        # compute a superpoints to superpixels loss using superpixels
        # torch.cuda.empty_cache()  # This method is extremely memory intensive
        superpixels = batch["superpixels"]  # batch中每个pixel的超像素id
        pairing_images = batch["pairing_images"]  # 点云点投影到image上对应的pixel坐标
        pairing_points = batch["pairing_points"]  # 能投影至图像的点的下标

        superpixels = (
                torch.arange(
                    0,
                    output_images.shape[0] * self.superpixel_size,
                    self.superpixel_size,  # 150
                    device=self.device,
                )[:, None, None] + superpixels
        )  # (12,224,416)
        m = tuple(pairing_images.cpu().T.long())  # pixel坐标

        superpixels_I = superpixels.flatten()
        idx_P = torch.arange(pairing_points.shape[0], device=superpixels.device)
        total_pixels = superpixels_I.shape[0]
        idx_I = torch.arange(total_pixels, device=superpixels.device)

        with torch.no_grad():
            one_hot_P = torch.sparse_coo_tensor(
                indices=torch.stack((
                    superpixels[m], idx_P
                ), dim=0),  #
                values=torch.ones(pairing_points.shape[0], device=superpixels.device),
                size=(superpixels.shape[0] * self.superpixel_size, pairing_points.shape[0])
            )  # 超像素与点的匹配对 (1800,30455) / (超像素的的个数,

            one_hot_I = torch.sparse_coo_tensor(
                indices=torch.stack((
                    superpixels_I, idx_I
                ), dim=0),
                values=torch.ones(total_pixels, device=superpixels.device),
                size=(superpixels.shape[0] * self.superpixel_size, total_pixels)
            )  # 超像素与图像的匹配对

        k = one_hot_P @ output_points[pairing_points]  # 超像素的投影点特征的求和; 超点的求和特征 [1800,512]
        k = k / (torch.sparse.sum(one_hot_P, 1).to_dense()[:, None] + 1e-6)  # 超像素的平均投影点特征； 超点的平均特征 [1800,512]
        q = one_hot_I @ output_images.permute(0, 2, 3, 1).flatten(0, 2)  # 超像素的图像特征的求和; 超像素的求和特征 [1800,512]
        q = q / (torch.sparse.sum(one_hot_I, 1).to_dense()[:, None] + 1e-6)  # 超像素的平均图像特征; 超像素的平均特征 [1800,512]

        loss1 = (1 - torch.nn.CosineSimilarity()(k, q)).mean()
        mask = torch.where(k[:, 0] != 0)  # 有投影点的超像素mask
        k = k[mask]  # 有投影点的超像素的点特征
        q = q[mask]  # 有投影点的超像素的图像特征

        loss = (1 - torch.nn.CosineSimilarity()(k, q)).mean()
        # print(loss1, loss)
        return loss

        # return self.criterion(k, q)

    def loss_superpixels_average_and_p2seg(self, batch, output_points, output_images):
        """
            计算point和segment之间的对比学习
        """
        torch.cuda.empty_cache()  # This method is extremely memory intensive
        superpixels = batch["superpixels"]  # batch中每个pixel的超像素id  (16*6,224,416)
        pairing_images = batch["pairing_images"]  # 点云点投影到image上对应的pixel坐标  (211471,3)
        pairing_points = batch["pairing_points"]  # 能投影至图像的点的下标  (211471,)

        superpixels = (
                torch.arange(
                    start=0,
                    end=output_images.shape[0] * self.superpixel_size,
                    step=self.superpixel_size,  # 150
                    device=self.device,
                )[:, None, None] + superpixels
        )  # [96,224,416]
        m = tuple(pairing_images.cpu().T.long())  # pixel坐标

        superpixels_I = superpixels.flatten()
        idx_P = torch.arange(pairing_points.shape[0], device=superpixels.device)
        total_pixels = superpixels_I.shape[0]
        idx_I = torch.arange(total_pixels, device=superpixels.device)

        with torch.no_grad():
            one_hot_P = torch.sparse_coo_tensor(
                indices=torch.stack((
                    superpixels[m], idx_P
                ), dim=0),  #
                values=torch.ones(pairing_points.shape[0], device=superpixels.device),
                size=(superpixels.shape[0] * self.superpixel_size, pairing_points.shape[0])
            )  # 超像素与点的匹配对 (24576,218051)

            one_hot_I = torch.sparse_coo_tensor(
                indices=torch.stack((
                    superpixels_I, idx_I
                ), dim=0),
                values=torch.ones(total_pixels, device=superpixels.device),
                size=(superpixels.shape[0] * self.superpixel_size, total_pixels)
            )  # 超像素与图像的匹配对 (24576,8945664)

        k = one_hot_P @ output_points[pairing_points]  # 超像素的投影点特征的求和/ 仅求outputs_point的   (7200,114854)
        k = k / (torch.sparse.sum(one_hot_P, 1).to_dense()[:, None] + 1e-6)  # 超像素的平均投影点特征
        q = one_hot_I @ output_images.permute(0, 2, 3, 1).flatten(0, 2)  # 超像素的图像特征的求和
        q = q / (torch.sparse.sum(one_hot_I, 1).to_dense()[:, None] + 1e-6)  # 超像素的平均图像特征

        mask = torch.where(k[:, 0] != 0)  # 有投影点的超像素mask
        k = k[mask]  # 有投影点的超像素的点特征    [3502,64]
        q = q[mask]  # 有投影点的超像素的图像特征  [3502,64]

        superpoint_superpixel_loss = self.criterion(k, q)

        #################
        # point_segment loss
        point_feats = output_points[pairing_points]  # point  (114854,64)
        superpoint_feats = k  # superpoint [3502, 64]
        dis_point_superpoint = torch.mm(point_feats, q.transpose(1, 0))  # (114854, 3502)
        # 每个point对应的segment
        target = one_hot_P.transpose(0, 1).coalesce().indices()[1, :]  # (114854,) 每个point所对应的segments id（未经过mask后的id）
        # mask[0][target]
        # a, b = torch.unique(mask[0], return_inverse=True)
        a = torch.sparse_coo_tensor(
            indices=torch.stack(mask, dim=0),
            values=torch.arange(superpoint_feats.shape[0], device=superpixels.device),
            size=(mask[0][-1] + 1,)
        ).to_dense()
        target_mapped = a[target].long()
        pixel_and_segment_loss = F.cross_entropy(dis_point_superpoint, target_mapped)

        # loss = superpoint_superpixel_loss+
        loss = superpoint_superpixel_loss + pixel_and_segment_loss
        return loss

    def training_epoch_end(self, outputs):
        self.epoch += 1
        if self.epoch == self.num_epochs:
            self.save()
        return super().training_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        sparse_input = ME.SparseTensor(batch["sinput_F"], batch["sinput_C"])
        output_points = self.model_points(sparse_input).F
        self.model_images.eval()
        output_images = self.model_images(batch["input_I"])

        if self._config['images_encoder'] == 'maskclip':
            output_images_feats, output_images_pred = output_images
            output_images = output_images_feats

        losses = [
            getattr(self, loss)(batch, output_points, output_images)
            for loss in self.losses
        ]
        loss = torch.mean(torch.stack(losses))
        self.val_losses.append(loss.detach().cpu())

        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size
        )
        return loss

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
