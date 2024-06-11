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
from MinkowskiEngine import SparseTensor
# from lightning.pytorch.utilities import
import yaml
from downstream.semantic_segmentation.criterion import DownstreamLoss
# from utils.metrics import confusion_matrix, comlpute_IoU_from_cmatrix
from utils.metrics import compute_IoU
import json

from utils.metrics import confusion_matrix, compute_IoU_from_cmatrix

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
CLASSES_SEM_STF = CLASSES_KITTI


class LightningDownstream(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.best_mIoU = -1.0
        self.metrics = {"val mIoU": [], "val_loss": [], "train_loss": []}
        self._config = config
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
                {"params": iter(param_group_head), "lr": self._config["lr_head"], "weight_decay": weight_decay_head},
                {"params": iter(param_group_trunk)}]
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

    def forward(self, x):
        return self.model(x).F

    def training_step(self, batch, batch_idx):
        if self._config["freeze_layers"]:
            self.model.eval()
        else:
            self.model.train()

        if '3d_model' in self._config and self._config['3d_model'] == 'MinkUnet18A':
            batch["sinput_F"] = torch.ones((batch["sinput_F"].shape[0], 3), device=batch["sinput_F"].device)
        sparse_input = SparseTensor(batch["sinput_F"], batch["sinput_C"])
        output_points = self(sparse_input)

        loss = self.criterion(output_points, batch["labels"])
        # torch.cuda.empty_cache() # empty the cache to reduce the memory requirement: ME is known to slowly
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
        if '3d_model' in self._config and self._config['3d_model'] == 'MinkUnet18A':
            batch["sinput_F"] = torch.ones((batch["sinput_F"].shape[0], 3), device=batch["sinput_F"].device)
        sparse_input = SparseTensor(batch["sinput_F"], batch["sinput_C"])
        output_points = self.model(sparse_input).F
        gt = batch['evaluation_labels']
        if self.ignore_index:
            output_points[:, self.ignore_index] = 0.0

        pred = output_points.argmax(1).cpu()

        offset = 0
        preds = []
        labels = []
        for j, lb in enumerate(batch["len_batch"]):
            inverse_indexes = batch["inverse_indexes"][j].cpu()
            predictions = pred[inverse_indexes + offset]

            # remove the ignored index entirely
            preds.append(predictions.detach().cpu())
            labels.append(gt[j].detach().cpu())
            offset += lb
        # self.val_full_predictions.extend(preds)
        # self.val_ground_truth.extend(labels)
        #####
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        c_matrix = confusion_matrix(preds, labels, self.n_classes)
        self.matrix.append(c_matrix)

        return None
        # return c_matrix

    def on_validation_epoch_end(self) -> None:
        c_matrix = sum(self.matrix)  #

        # remove the ignore_index from the confusion matrix
        if self.trainer.num_devices > 1:
            c_matrix = torch.sum(self.all_gather(c_matrix), 0)

        m_IoU, fw_IoU, per_class_IoU = compute_IoU_from_cmatrix(
            c_matrix.cpu(), self.ignore_index
        )

        best_epoch = False
        if self.best_mIoU < m_IoU:
            self.best_mIoU = m_IoU
            best_epoch = True

        results_dict = {
            "mean_IoU": m_IoU,
            "epoch": self.current_epoch
        }

        if self._config["dataset"].lower() == "nuscenes":
            for a, b in zip(CLASSES_NUSCENES, (per_class_IoU).numpy()):
                results_dict[a] = float(b)
        elif self._config["dataset"].lower() == "kitti":
            for a, b in zip(CLASSES_KITTI, (per_class_IoU).numpy()):
                results_dict[a] = float(b)
        elif self._config['dataset'].lower() == 'semanticstf':
            for a, b in zip(CLASSES_SEM_STF, (per_class_IoU).numpy()):
                results_dict[a] = float(b)
        else:
            raise ValueError(f"Unknown dataset : {self._config['dataset']}")

        if best_epoch:
            self.save(results_dict, best=True)

        self.log_dict(
            {
                "mIoU": m_IoU,
                "fw_IoU": fw_IoU,
                "best_mIoU": self.best_mIoU,
            },
            on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )

        if self.epoch >= self._config["num_epochs"] - 1:
            self.save(results_dict, best=False)

    @rank_zero_only
    def save(self, results_dict, best=False):
        # print(results_dict)
        if best:
            path = os.path.join(self.working_dir, f"best_model.pt")
        else:
            path = os.path.join(self.working_dir, f"model.pt")

        torch.save(
            {
                "model_points": self.model.state_dict(),
                "config": self._config
            },
            path
        )
        ####
        results_json = json.dumps(results_dict, sort_keys=False)
        if best:
            json_path = "best_results.json"
        else:
            json_path = "results.json"
        with open(os.path.join(self.working_dir, json_path), 'w') as f:
            f.write(results_json)
