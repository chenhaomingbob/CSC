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
# SemanticKITTI
from downstream.semantic_segmentation.dataloader_kitti import SemanticKITTIDataset
from downstream.semantic_segmentation.dataloader_kitti import \
    minkunet_custom_collate_fn as sk_minkunet_custom_collate_fn
from downstream.semantic_segmentation.dataloader_kitti_spconv import SemanticKITTIDatasetSpconv
from downstream.semantic_segmentation.dataloader_kitti_spconv import \
    spconv_custom_collate_fn as sk_spconv_custom_collate_fn
# nuScenes
from downstream.semantic_segmentation.dataloader_nuscenes import NuScenesDataset
from downstream.semantic_segmentation.dataloader_nuscenes import \
    minkunet_custom_collate_fn as nu_minkunet_custom_collate_fn
from downstream.semantic_segmentation.dataloader_nuscenes_spconv import NuScenesDatasetSpconv
from downstream.semantic_segmentation.dataloader_nuscenes_spconv import \
    spconv_custom_collate_fn as nu_spconv_custom_collate_fn
# SemanticSTF
from downstream.semantic_segmentation.dataloader_semantic_stf import SemanticSTFDataset
from downstream.semantic_segmentation.dataloader_semantic_stf import \
    minkunet_custom_collate_fn as sem_stf_custom_collate_fn


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
        self.collate_fn = None

    def setup(self, stage):
        # setup the dataloader: this function is automatically called by lightning
        transforms = make_transforms_clouds(self.config)
        dataset = self.config['dataset']
        model_points = self.config['model_points']  # 'minkunet' or 'voxelnet'
        dataset_fn, self.collate_fn = get_dataset_collate_fn(dataset, model_points)

        if self.config["training"] in ("parametrize", "parametrizing"):
            phase_train = "parametrizing"
            phase_val = "verifying"
        else:
            phase_train = "train"
            phase_val = "val"

        # train dataset
        self.train_dataset = dataset_fn(
            phase=phase_train,
            transforms=transforms,
            config=self.config
        )

        # validation dataset
        if dataset.lower() == "nuscenes":
            self.val_dataset = dataset_fn(
                phase=phase_val,
                config=self.config,
                cached_nuscenes=self.train_dataset.nusc,
            )
        else:
            self.val_dataset = dataset_fn(
                phase=phase_val,
                config=self.config
            )

    def train_dataloader(self):
        # construct the training dataloader: this function is automatically called
        # by lightning
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=lambda id: np.random.seed(
                torch.initial_seed() // 2 ** 32 + id
            ),
        )

    def val_dataloader(self):
        # construct the validation dataloader: this function is automatically called
        # by lightning
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=lambda id: np.random.seed(
                torch.initial_seed() // 2 ** 32 + id
            ),
        )


def get_dataset_collate_fn(dataset_name, model_points_name):
    dataset_name = dataset_name.lower()
    model_points_name = model_points_name.lower()
    dataset_fn, collate_fn = None, None

    assert dataset_name in ['nuscenes', 'semantickitti', 'kitti', 'semanticstf']
    assert model_points_name in ['minkunet', 'voxelnet', 'cylinder3d']
    if dataset_name == "nuscenes":
        if model_points_name == 'minkunet':
            dataset_fn = NuScenesDataset
            collate_fn = nu_minkunet_custom_collate_fn
        elif model_points_name == 'voxelnet':
            dataset_fn = NuScenesDatasetSpconv
            collate_fn = nu_spconv_custom_collate_fn
    elif dataset_name in ("kitti", "semantickitti"):
        if model_points_name == 'minkunet':
            dataset_fn = SemanticKITTIDataset
            collate_fn = sk_minkunet_custom_collate_fn
        elif model_points_name == 'voxelnet':
            dataset_fn = SemanticKITTIDatasetSpconv
            collate_fn = sk_spconv_custom_collate_fn
    elif dataset_name in ('semanticstf'):
        if model_points_name == 'minkunet':
            dataset_fn = SemanticSTFDataset
            collate_fn = sem_stf_custom_collate_fn
        elif model_points_name == 'voxelnet':
            raise ValueError("Not Support voxelnet")
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    return dataset_fn, collate_fn
