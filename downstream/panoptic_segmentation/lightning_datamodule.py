import torch
import numpy as np
import pytorch_lightning

if pytorch_lightning.__version__.startswith("1"):
    import pytorch_lightning as pl
    from pytorch_lightning.utilities import rank_zero_only
elif pytorch_lightning.__version__.startswith("2"):
    import lightning.pytorch as pl
    from lightning.pytorch.utilities import rank_zero_only
from torch.utils.data import DataLoader
from utils.transforms import make_transforms_clouds
from downstream.panoptic_segmentation.dataloader_kitti import SemanticKITTIDataset
from downstream.panoptic_segmentation.dataloader_nuscenes import NuScenesDataset, minkunet_custom_collate_fn
from downstream.panoptic_segmentation.dataloader_nuscenes_spconv import NuScenesDatasetSpconv, spconv_custom_collate_fn


class DownstreamDataModule(pl.LightningDataModule):
    """
    The equivalent of a DataLoader for pytorch lightning.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        # in multi-GPU the actual batch size is that
        self.batch_size = config["batch_size"] // config["num_gpus"]
        # the CPU workers are split across GPU
        self.num_workers = max(config["num_threads"] // config["num_gpus"], 1)

        self.model_points_name = config['model_points'].lower()

    def setup(self, stage):
        # setup the dataloader: this function is automatically called by lightning
        transforms = make_transforms_clouds(self.config)

        if self.config["dataset"].lower() == "nuscenes":
            if self.model_points_name == 'minkunet':
                Dataset = NuScenesDataset
            else:
                Dataset = NuScenesDatasetSpconv
        elif self.config["dataset"].lower() in ("kitti", "semantickitti"):
            if self.model_points_name == 'minkunet':
                Dataset = SemanticKITTIDataset
            else:
                Dataset = SemanticKITTIDatasetSpconv
        else:
            raise Exception(f"Unknown dataset {self.config['dataset']}")

        if self.config["training"] in ("parametrize", "parametrizing"):
            phase_train = "parametrizing"
            phase_val = "verifying"
        else:
            phase_train = "train"
            phase_val = "val"

        self.train_dataset = Dataset(
            phase=phase_train, transforms=transforms, config=self.config
        )
        if Dataset == NuScenesDataset or Dataset == NuScenesDatasetSpconv:
            self.val_dataset = Dataset(
                phase=phase_val,
                config=self.config,
                cached_nuscenes=self.train_dataset.nusc,
            )
        else:
            self.val_dataset = Dataset(phase=phase_val, config=self.config)

    def train_dataloader(self):
        # construct the training dataloader: this function is automatically called
        # by lightning
        # if self.config["num_gpus"]:
        #     num_workers = self.config["num_threads"] // self.config["num_gpus"]
        # else:
        #     num_workers = self.config["num_threads"]

        if self.config["model_points"] == "minkunet":
            default_collate_pair_fn = minkunet_custom_collate_fn
        else:
            default_collate_pair_fn = spconv_custom_collate_fn

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=default_collate_pair_fn,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=lambda id: np.random.seed(
                torch.initial_seed() // 2 ** 32 + id
            ),
        )

    def val_dataloader(self):
        # construct the validation dataloader: this function is automatically called
        # by lightning
        # if self.config["num_gpus"]:
        #     num_workers = self.config["num_threads"] // self.config["num_gpus"]
        # else:
        #     num_workers = self.config["num_threads"]
        #
        print(self.num_workers)
        if self.config["model_points"] == "minkunet":
            default_collate_pair_fn = minkunet_custom_collate_fn
        else:
            default_collate_pair_fn = spconv_custom_collate_fn

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=default_collate_pair_fn,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=lambda id: np.random.seed(
                torch.initial_seed() // 2 ** 32 + id
            ),
        )
