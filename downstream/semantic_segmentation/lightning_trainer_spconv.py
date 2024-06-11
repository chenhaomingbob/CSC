import os
from copy import deepcopy
import pytorch_lightning

if pytorch_lightning.__version__.startswith("1"):
    import pytorch_lightning as pl
    from pytorch_lightning.utilities import rank_zero_only
elif pytorch_lightning.__version__.startswith("2"):
    import lightning.pytorch as pl
    from lightning.pytorch.utilities import rank_zero_only
import torch
import torch.optim as optim
# from lightning.pytorch.utilities import
import yaml
from downstream.semantic_segmentation.criterion import DownstreamLoss
# from utils.metrics import confusion_matrix, comlpute_IoU_from_cmatrix
from utils.metrics import compute_IoU
import json
import numpy as np
from utils.metrics import confusion_matrix, compute_IoU_from_cmatrix
from spconv.pytorch.utils import gather_features_by_pc_voxel_id

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


class LightningDownstreamSpconv(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.best_mIoU = 0.0
        self.metrics = {"val mIoU": [], "val_loss": [], "train_loss": []}
        self._config = config
        self.batch_size = config["batch_size"]
        self.train_losses = []
        self.val_losses = []
        self.ignore_index = config["ignore_index"]
        self.n_classes = config["model_n_out"]
        self.epoch = 0
        if config["loss"].lower() == "lovasz":
            self.criterion = DownstreamLoss(
                ignore_index=config["ignore_index"],
                device=self.device,
            )  # a lovasz loss and a crossentropy
        else:
            self.criterion = torch.nn.CrossEntropyLoss(
                ignore_index=config["ignore_index"],
            )

        self.working_dir = config['working_path']
        self.make_dir()
        self.save_hyperparameters('config')
        self.validation_step_outputs = []

        self.ignore_index = self._config['ignore_index']

        self.val_full_predictions = []
        self.val_ground_truth = []

    @rank_zero_only
    def make_dir(self):
        os.makedirs(self.working_dir, exist_ok=True)
        config_save_path = os.path.join(self.working_dir, 'config_file.yaml')
        with open(config_save_path, 'w') as f:
            yaml.dump(self._config, f)
        print(f"working_dir {os.path.abspath(self.working_dir)}")

    def configure_optimizers(self):
        if self._config.get("lr_head", None) is not None:
            print("Use different learning rates between the head and trunk.")

            def is_final_head(key):
                return key.find('final.') != -1

            param_group_head = [
                param for key, param in self.model.named_parameters()
                if param.requires_grad and is_final_head(key)]
            param_group_trunk = [
                param for key, param in self.model.named_parameters()
                if param.requires_grad and (not is_final_head(key))]
            param_group_all = [
                param for key, param in self.model.named_parameters()
                if param.requires_grad]
            assert len(param_group_all) == (len(param_group_head) + len(param_group_trunk))

            weight_decay = self._config["weight_decay"]
            weight_decay_head = self._config["weight_decay_head"] if (
                    self._config["weight_decay_head"] is not None) else weight_decay
            parameters = [
                {
                    "params": iter(param_group_head),
                    "lr": self._config["lr_head"],
                    "weight_decay": weight_decay_head
                },
                {
                    "params": iter(param_group_trunk)
                }
            ]
            print(
                f"==> Head:  #{len(param_group_head)} params with learning rate: {self._config['lr_head']} and weight_decay: {weight_decay_head}")
            print(
                f"==> Trunk: #{len(param_group_trunk)} params with learning rate: {self._config['lr']} and weight_decay: {weight_decay}")

            optimizer = optim.SGD(
                parameters,
                lr=self._config["lr"],
                momentum=self._config["sgd_momentum"],
                dampening=self._config["sgd_dampening"],
                weight_decay=self._config["weight_decay"],
            )
        else:
            if self._config.get("optimizer") and self._config["optimizer"] == 'adam':
                print('Optimizer: AdamW')
                optimizer = optim.AdamW(
                    self.model.parameters(),
                    lr=self._config["lr"],
                    weight_decay=self._config["weight_decay"],
                )
            else:
                print('Optimizer: SGD')
                optimizer = optim.SGD(
                    self.model.parameters(),
                    lr=self._config["lr"],
                    momentum=self._config["sgd_momentum"],
                    dampening=self._config["sgd_dampening"],
                    weight_decay=self._config["weight_decay"],
                )

        if self._config.get("scheduler") and self._config["scheduler"] == 'steplr':
            print('Scheduler: StepLR')
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, int(.9 * self._config["num_epochs"]),
            )
        else:
            print('Scheduler: Cosine')
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self._config["num_epochs"]
            )
        return [optimizer], [scheduler]

    # def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
    #     # set_to_none=True is a modest speed-up
    #     optimizer.zero_grad(set_to_none=True)

    def forward(self, x):
        # print(self.model(x))
        return self.model(x).F

    def training_step(self, batch, batch_idx):
        if self._config["freeze_layers"]:
            self.model.eval()
        else:
            self.model.train()

        output_points = self.model(batch["voxels"], batch["coordinates"])
        output_points = interpolate_from_bev_features(
            batch["pc"],
            output_points,
            self.batch_size,
            self.model.bev_stride
        )

        loss = self.criterion(output_points, batch["labels"])

        self.log(
            "loss", loss,
            on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(batch['pc']), sync_dist=True
        )
        self.train_losses.append(loss.detach().cpu())
        return loss

    def on_train_epoch_end(self) -> None:
        self.epoch += 1

    def on_validation_epoch_start(self) -> None:
        self.matrix = []

    def validation_step(self, batch, batch_idx):
        output_points = self.model(batch["voxels"], batch["coordinates"])  # (32,17,128,128)
        output_points = interpolate_from_bev_features(
            batch["pc"],
            output_points,
            self.batch_size,
            self.model.bev_stride
        )
        # 将voxel prediction 转换成 point prediction
        gt = batch['evaluation_labels']
        pc_voxel_id = batch['pc_voxel_id']  # list[Tensor]
        len_pc_batch = batch['len_batch']  # list[int]
        if self.ignore_index:
            output_points[:, self.ignore_index] = 0.0

        pred = output_points.argmax(1).cpu()  #

        offset = 0
        preds = []
        labels = []
        for index, num_point in enumerate(len_pc_batch):
            cur_pc_voxel_id = pc_voxel_id[index]
            cur_voxel_predictions = pred[offset:offset + num_point]
            cur_point_pred = gather_features_by_pc_voxel_id(cur_voxel_predictions, cur_pc_voxel_id)
            #
            preds.append(cur_point_pred.detach().cpu())
            labels.append(gt[index].detach().cpu())
            offset += num_point

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        c_matrix = confusion_matrix(preds, labels, self.n_classes)
        self.matrix.append(c_matrix)

        return None

    def on_validation_epoch_end(self) -> None:
        c_matrix = sum(self.matrix)  #

        # remove the ignore_index from the confusion matrix
        if self.trainer.num_nodes > 1:
            c_matrix = torch.sum(self.all_gather(c_matrix), 0)

        m_IoU, fw_IoU, per_class_IoU = compute_IoU_from_cmatrix(
            c_matrix.cpu(), self.ignore_index
        )

        results_dict = {}
        if self._config["dataset"].lower() == "nuscenes":
            for a, b in zip(CLASSES_NUSCENES, (per_class_IoU).numpy()):
                results_dict[a] = float(b)
        elif self._config["dataset"].lower() == "kitti":
            for a, b in zip(CLASSES_KITTI, (per_class_IoU).numpy()):
                results_dict[a] = float(b)
        results_dict['mean_IoU'] = m_IoU
        results_dict['epoch'] = self.epoch
        if self.best_mIoU < m_IoU:
            self.best_mIoU = m_IoU
            self.save(results_dict, best=True)

        self.log_dict(
            {
                "val_m_IoU": m_IoU,
                "val_fw_IoU": fw_IoU,
                "best_m_IoU": self.best_mIoU,
            },
            on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        if self.epoch == self._config["num_epochs"]:
            self.save(results_dict, best=False)

    @rank_zero_only
    def save(self, results_dict, best=False):
        if best:
            path = os.path.join(self.working_dir, f"best_model.pt")
        else:
            path = os.path.join(self.working_dir, f"model.pt")

        torch.save(
            {"model_points": self.model.state_dict(), "config": self._config}, path
        )
        ####
        results_json = json.dumps(results_dict, sort_keys=False)
        if best:
            json_path = "best_results.json"
        else:
            json_path = "results.json"
        with open(os.path.join(self.working_dir, json_path), 'w') as f:
            f.write(results_json)


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
