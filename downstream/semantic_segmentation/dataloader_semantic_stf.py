import os
import os.path as osp
import re
import torch
import numpy as np
from torch.utils.data import Dataset
from MinkowskiEngine.utils import sparse_quantize
from utils.transforms import make_transforms_clouds

TRAIN_SET = ['train']
VAL_SET = ['val']
TEST_SET = ['test']

stf_pc_data_file_suffix = 'bin'
stf_pc_data_directory_name = 'velodyne'
stf_pc_label_file_suffix = 'label'
stf_dataset_root_dir = './datasets/SemanticSTF/raw'

stf_pc_label_directory_name = 'labels'  # for

learning_map = {
    0: 255,  # "unlabeled",
    1: 0,  # "car",
    2: 1,  # "bicycle",
    3: 2,  # "motorcycle",
    4: 3,  # "truck",
    5: 4,  # "other-vehicle",
    6: 5,  # "person",
    7: 6,  # "bicyclist",
    8: 7,  # "motorcyclist",
    9: 8,  # "road",
    10: 9,  # "parking",
    11: 10,  # "sidewalk",
    12: 11,  # "other-ground",
    13: 12,  # "building",
    14: 13,  # "fence",
    15: 14,  # "vegetation",
    16: 15,  # "trunk",
    17: 16,  # "terrain",
    18: 17,  # "pole",
    19: 18,  # "traffic-sign",
    20: 255  # "invalid"
}

learning_map_inv = {  # inverse of previous map
    255: 0,  # "unlabeled", and others ignored
    0: 1,  # "car"
    1: 2,  # "bicycle"
    2: 3,  # "motorcycle"
    3: 4,  # "truck"
    4: 5,  # "other-vehicle"
    5: 6,  # "person"
    6: 7,  # "bicyclist"
    7: 8,  # "motorcyclist"
    8: 9,  # "road"
    9: 10,  # "parking"
    10: 11,  # "sidewalk"
    11: 12,  # "other-ground"
    12: 13,  # "building"
    13: 14,  # "fence"
    14: 15,  # "vegetation"
    15: 16,  # "trunk"
    16: 17,  # "terrain"
    17: 18,  # "pole"
    18: 19  # "traffic-sign"
}


def minkunet_custom_collate_fn(list_data):
    """
    Collate function adapted for creating batches with MinkowskiEngine.
    """
    input = list(zip(*list_data))
    labelized = len(input) == 6
    if labelized:
        xyz, coords, feats, labels, evaluation_labels, inverse_indexes = input
    else:
        xyz, coords, feats, inverse_indexes = input

    coords_batch, len_batch = [], []

    for batch_id, coo in enumerate(coords):
        N = coords[batch_id].shape[0]
        coords_batch.append(
            torch.cat((torch.ones(N, 1, dtype=torch.int32) * batch_id, coo), 1)
        )
        len_batch.append(N)

    # Concatenate all lists
    coords_batch = torch.cat(coords_batch, 0).int()
    feats_batch = torch.cat(feats, 0).float()
    if labelized:
        labels_batch = torch.cat(labels, 0).long()
        return {
            "pc": xyz,  # point cloud
            "sinput_C": coords_batch,  # discrete coordinates (ME)
            "sinput_F": feats_batch,  # point features (N, 3)
            "len_batch": len_batch,  # length of each batch
            "labels": labels_batch,  # labels for each (voxelized) point
            "evaluation_labels": evaluation_labels,  # labels for each point
            "inverse_indexes": inverse_indexes,  # labels for each point
        }
    else:
        return {
            "pc": xyz,
            "sinput_C": coords_batch,
            "sinput_F": feats_batch,
            "len_batch": len_batch,
            "inverse_indexes": inverse_indexes,
        }


class SemanticSTFDataset(Dataset):
    """
    Dataset returning a lidar scene and associated labels.
    Note that superpixels fonctionality have been removed.
    """

    def __init__(self, phase, config, transforms=None):
        ##################################################################
        self.phase = phase
        self.labels = self.phase != "test"
        self.transforms = transforms
        self.voxel_size = config["voxel_size"]
        self.cylinder = config["cylindrical_coordinates"]

        if not os.path.exists(stf_dataset_root_dir):
            raise Exception(f"Check dataset directory: {os.path.abspath(stf_dataset_root_dir)}")

        ##################################################################
        # a skip ratio can be used to reduce the dataset size
        # and accelerate experiments
        if phase == "train":
            try:
                skip_ratio = config["dataset_skip_step"]
            except KeyError:
                skip_ratio = 1
        else:
            skip_ratio = 1

        if phase == 'train':
            phase_set = TRAIN_SET
        elif phase == 'val':
            phase_set = VAL_SET
        elif phase == 'test':
            phase_set = TEST_SET

        self.list_files = []

        for seq in phase_set:
            seq_files = sorted(
                os.listdir(os.path.join(stf_dataset_root_dir, seq, stf_pc_data_directory_name)))
            seq_files = [
                os.path.join(stf_dataset_root_dir, seq, stf_pc_data_directory_name, x) for x in seq_files
            ]
            self.list_files.extend(seq_files)

        self.list_files = sorted(self.list_files)[::skip_ratio]
        #####
        remap_dict = learning_map
        max_key = max(remap_dict.keys())
        remap_lut = np.ones((max_key + 100), dtype=np.int32) * 255
        remap_lut[list(remap_dict.keys())] = list(remap_dict.values())
        self.label_map = remap_lut

        self.eval_labels = {
            255: 0,  # "unlabeled", and others ignored
            0: 1,  # "car"
            1: 2,  # "bicycle"
            2: 3,  # "motorcycle"
            3: 4,  # "truck"
            4: 5,  # "other-vehicle"
            5: 6,  # "person"
            6: 7,  # "bicyclist"
            7: 8,  # "motorcyclist"
            8: 9,  # "road"
            9: 10,  # "parking"
            10: 11,  # "sidewalk"
            11: 12,  # "other-ground"
            12: 13,  # "building"
            13: 14,  # "fence"
            14: 15,  # "vegetation"
            15: 16,  # "trunk"
            16: 17,  # "terrain"
            17: 18,  # "pole"
            18: 19  # "traffic-sign"
        }

        self.reverse_label_name_mapping = learning_map_inv
        self.num_classes = 19
        self.angle = 0.0

    def set_angle(self, angle):
        self.angle = angle

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        lidar_file = self.list_files[idx]
        with open(lidar_file, 'rb') as b:
            block_ = np.fromfile(b, dtype=np.float32).reshape(-1, 5)[:, :4]
            block_[:, 3] /= 255.
        pc = block_[:, :3]

        if self.labels:
            label_file = lidar_file.replace(stf_pc_data_directory_name, stf_pc_label_directory_name).replace(
                stf_pc_data_file_suffix, stf_pc_label_file_suffix)

            if os.path.exists(label_file):
                with open(label_file, 'rb') as a:
                    all_labels = np.fromfile(a, dtype=np.int32).reshape(-1)
            else:
                all_labels = np.zeros(pc.shape[0]).astype(np.int32)

            points_labels = self.label_map[all_labels].astype(np.int64)

        pc = torch.tensor(pc)

        # apply the transforms (augmentation)
        if self.transforms:
            pc = self.transforms(pc)

        if self.cylinder:
            # Transform to cylinder coordinate and scale for voxel size
            x, y, z = pc.T
            rho = torch.sqrt(x ** 2 + y ** 2) / self.voxel_size
            # corresponds to a split each 1°
            phi = torch.atan2(y, x) * 180 / np.pi
            z = z / self.voxel_size
            coords_aug = torch.cat((rho[:, None], phi[:, None], z[:, None]), 1)
        else:
            coords_aug = pc / self.voxel_size

        # Voxelization
        discrete_coords, indexes, inverse_indexes = sparse_quantize(
            coords_aug, return_index=True, return_inverse=True
        )
        unique_feats = torch.tensor(block_[indexes][:, 3:] + 1.)

        if self.labels:
            points_labels = torch.tensor(
                np.vectorize(self.eval_labels.__getitem__)(points_labels),
                dtype=torch.int32,
            )
            unique_labels = points_labels[indexes]

        if self.labels:
            return (
                pc,
                discrete_coords,
                unique_feats,
                unique_labels,
                points_labels,
                inverse_indexes,
            )
        else:
            return pc, discrete_coords, unique_feats, inverse_indexes


def make_data_loader_semstf(config, phase, num_threads=0):
    """
    Create the data loader for a given phase and a number of threads.
    """
    # select the desired transformations
    if phase == "train":
        transforms = make_transforms_clouds(config)
    else:
        transforms = None

    # instantiate the dataset
    dset = SemanticSTFDataset(phase=phase, transforms=transforms, config=config)
    collate_fn = minkunet_custom_collate_fn

    batch_size = config["batch_size"] // config["num_gpus"]

    # create the loader
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        # shuffle=False if sampler else True,
        shuffle=phase == "train",
        num_workers=num_threads,
        collate_fn=collate_fn,
        pin_memory=False,
        # sampler=sampler,
        drop_last=phase == "train",
        worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
    )
    return loader
