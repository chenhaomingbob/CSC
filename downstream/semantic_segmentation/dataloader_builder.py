import torch
import numpy as np
from downstream.semantic_segmentation.lightning_datamodule import get_dataset_collate_fn
from utils.transforms import make_transforms_clouds

TRAIN_PHASE = 'train'
VAL_PHASE = 'val'
TEST_PHASE = 'test'


def make_data_loader_nuscenes(config):
    """
    Create the data loader for a given phase and a number of threads.
    This function is not used with pytorch lightning, but is used when evaluating.
    """
    # select the desired transformations
    phase = config['data_split']
    num_threads = config['num_threads']
    dataset_name = config['dataset']
    if 'backbone' in config:
        model_points_name = config['backbone']
    elif 'model_points' in config:
        model_points_name = config['model_points']
    else:
        raise Exception("'backbone' or 'model_points' not in the key list of config")
    if phase == TRAIN_PHASE:
        transforms = make_transforms_clouds(config)
    else:
        transforms = None

    dataset_fn, collate_fn = get_dataset_collate_fn(dataset_name=dataset_name, model_points_name=model_points_name)

    dataset = dataset_fn(phase=phase, transforms=transforms, config=config)

    batch_size = config["batch_size"] // config["num_gpus"]

    # create the loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=phase == TRAIN_PHASE,
        num_workers=num_threads,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=phase == TRAIN_PHASE,
        worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
    )
    return loader


def make_dataloader(config):
    """
    Build the points model according to what is in the config
    """

    dataloader = make_data_loader_nuscenes(config)

    return dataloader
