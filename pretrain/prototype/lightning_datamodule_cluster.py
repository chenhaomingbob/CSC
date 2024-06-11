import pytorch_lightning

if pytorch_lightning.__version__.startswith("1"):
    import pytorch_lightning as pl
    from pytorch_lightning.utilities import rank_zero_only
elif pytorch_lightning.__version__.startswith("2"):
    import lightning.pytorch as pl
    from lightning.pytorch.utilities import rank_zero_only
import numpy as np
import torch
from torch.utils.data import DataLoader

from pretrain.prototype.dataloader_nuscenes_cluster import (
    NuScenesMatchDataset,
    minkunet_collate_pair_fn)
from pretrain.prototype.dataloader_nuscenes_cluster_spconv import NuScenesMatchDatasetSpconv, spconv_collate_pair_fn
from pretrain.prototype.dataloader_nuscenes_cluster_cylinder3d \
    import NuScenesDatasetCylinder3d, cylinder3d_custom_collate_fn
# from pretrain.dataloader_nuscenes_spconv import NuScenesMatchDatasetSpconv, spconv_collate_pair_fn
from utils.transforms import (
    make_transforms_clouds,
    make_transforms_asymmetrical,
    make_transforms_asymmetrical_val,
)


class PretrainDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config["num_gpus"]:
            self.batch_size = config["batch_size"] // config["num_gpus"]
        else:
            self.batch_size = config["batch_size"]

    def setup(self, stage):
        cloud_transforms_train = make_transforms_clouds(self.config)
        mixed_transforms_train = make_transforms_asymmetrical(self.config)
        cloud_transforms_val = None
        mixed_transforms_val = make_transforms_asymmetrical_val(self.config)
        #####
        dataset_name = self.config["dataset"].lower()
        model_points_name = self.config["model_points"]

        if dataset_name == "nuscenes" and model_points_name == "minkunet":
            Dataset = NuScenesMatchDataset
        elif dataset_name == "nuscenes" and model_points_name == "voxelnet":
            Dataset = NuScenesMatchDatasetSpconv
        elif dataset_name == "nuscenes" and model_points_name in ("cylinder3d", "cylinder3d_separate"):
            Dataset = NuScenesDatasetCylinder3d
        else:
            raise Exception("Dataset Unknown")

        # ---------------------------------------------------------------------------
        phase_train = "train"
        phase_val = "cluster"  # 也用数据集

        self.train_dataset = Dataset(
            phase=phase_train,
            shuffle=True,
            cloud_transforms=cloud_transforms_train,
            mixed_transforms=mixed_transforms_train,
            config=self.config,
        )

        self.val_dataset = Dataset(
            phase=phase_val,  #
            shuffle=False,
            cloud_transforms=cloud_transforms_val,
            mixed_transforms=mixed_transforms_val,
            config=self.config,
            cached_nuscenes=self.train_dataset.nusc
        )

        print("Dataset Loaded")

    def train_dataloader(self):

        if self.config["num_gpus"]:
            num_workers = self.config["num_threads"] // self.config["num_gpus"]
        else:
            num_workers = self.config["num_threads"]

        if self.config["model_points"] == "minkunet":
            default_collate_pair_fn = minkunet_collate_pair_fn
        elif self.config["model_points"] == "voxelnet":
            default_collate_pair_fn = spconv_collate_pair_fn
        elif self.config["model_points"] in ("cylinder3d", "cylinder3d_separate"):
            default_collate_pair_fn = cylinder3d_custom_collate_fn
        else:
            raise Exception("3d Model Unknown")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=default_collate_pair_fn,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=lambda id: np.random.seed(
                torch.initial_seed() // 2 ** 32 + id
            ),
        )

    def val_dataloader(self):

        if self.config["num_gpus"]:
            num_workers = self.config["num_threads"] // self.config["num_gpus"]
        else:
            num_workers = self.config["num_threads"]

        if self.config["model_points"] == "minkunet":
            default_collate_pair_fn = minkunet_collate_pair_fn
        elif self.config["model_points"] == "voxelnet":
            default_collate_pair_fn = spconv_collate_pair_fn
        elif self.config["model_points"] in ("cylinder3d", "cylinder3d_separate"):
            default_collate_pair_fn = cylinder3d_custom_collate_fn
        else:
            raise Exception("3d Model Unknown")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=default_collate_pair_fn,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=lambda id: np.random.seed(
                torch.initial_seed() // 2 ** 32 + id
            ),
        )
