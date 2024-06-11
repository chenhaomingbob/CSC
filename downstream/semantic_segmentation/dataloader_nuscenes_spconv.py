import os

import numpy as np
import torch
from MinkowskiEngine.utils import sparse_quantize
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.splits import create_splits_scenes
from torch.utils.data import Dataset

from utils.transforms import make_transforms_clouds
from spconv.pytorch.utils import PointToVoxel

# parametrizing set, to try out different parameters
CUSTOM_SPLIT = [
    "scene-0008", "scene-0009", "scene-0019", "scene-0029", "scene-0032", "scene-0042",
    "scene-0045", "scene-0049", "scene-0052", "scene-0054", "scene-0056", "scene-0066",
    "scene-0067", "scene-0073", "scene-0131", "scene-0152", "scene-0166", "scene-0168",
    "scene-0183", "scene-0190", "scene-0194", "scene-0208", "scene-0210", "scene-0211",
    "scene-0241", "scene-0243", "scene-0248", "scene-0259", "scene-0260", "scene-0261",
    "scene-0287", "scene-0292", "scene-0297", "scene-0305", "scene-0306", "scene-0350",
    "scene-0352", "scene-0358", "scene-0361", "scene-0365", "scene-0368", "scene-0377",
    "scene-0388", "scene-0391", "scene-0395", "scene-0413", "scene-0427", "scene-0428",
    "scene-0438", "scene-0444", "scene-0452", "scene-0453", "scene-0459", "scene-0463",
    "scene-0464", "scene-0475", "scene-0513", "scene-0533", "scene-0544", "scene-0575",
    "scene-0587", "scene-0589", "scene-0642", "scene-0652", "scene-0658", "scene-0669",
    "scene-0678", "scene-0687", "scene-0701", "scene-0703", "scene-0706", "scene-0710",
    "scene-0715", "scene-0726", "scene-0735", "scene-0740", "scene-0758", "scene-0786",
    "scene-0790", "scene-0804", "scene-0806", "scene-0847", "scene-0856", "scene-0868",
    "scene-0882", "scene-0897", "scene-0899", "scene-0976", "scene-0996", "scene-1012",
    "scene-1015", "scene-1016", "scene-1018", "scene-1020", "scene-1024", "scene-1044",
    "scene-1058", "scene-1094", "scene-1098", "scene-1107",
]


def mean_vfe(voxel_features, voxel_num_points):
    # voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
    points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
    normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)  # 确保最小值为1.0
    points_mean = points_mean / normalizer
    voxel_features = points_mean.contiguous()

    return voxel_features


def spconv_custom_collate_fn(list_data):
    """
    Custom collate function adapted for creating batches with spconv.
    """
    input = list(zip(*list_data))

    labelized = len(input) >= 6

    if labelized:
        # 7 items
        xyz, coords, feats, labels, evaluation_labels, pc_voxel_id, num_points = input
    else:
        # 5 items
        xyz, coords, feats, pc_voxel_id, num_points = input

    len_batch, pc_batch = [], []

    for batch_id, coo in enumerate(coords):
        # Move batchids to the beginning
        coords[batch_id][:, 0] = batch_id
        len_batch.append(coords[batch_id].shape[0])  # 当前batch_id 点云的点数量
        pc_batch.append(
            torch.cat(
                [torch.ones((xyz[batch_id].shape[0], 1)) * batch_id, xyz[batch_id]], dim=1
            )
        )
    #################################
    ##### Concatenate all lists #####
    coords_batch = torch.cat(coords, 0).int()
    num_points = torch.cat(num_points, 0)
    pc_batch = torch.cat(pc_batch, 0)

    # voxel features
    feats_batch = torch.cat(feats, 0).float()  # (num_voxels, max_number_in_voxel, 4). e.g., (217827,10,4)
    feats_batch = mean_vfe(feats_batch, num_points)  # (num_voxels, 4). e.g., (217827,4)


    if labelized:
        labels_batch = torch.cat(labels, 0).long()

        return {
            "pc": pc_batch,  # point cloud
            "coordinates": coords_batch,
            "voxels": feats_batch,  # voxel features
            "num_points": num_points,  # the number of points for each voxel
            "labels": labels_batch,  # labels for each (voxelized) point
            "evaluation_labels": evaluation_labels,  # sem labels for each point
            "pc_voxel_id": pc_voxel_id,  # voxel id for each point
            "len_batch": len_batch
            #######
        }
    else:
        return {
            "pc": pc_batch,  # point cloud
            "coordinates": coords_batch,
            "voxels": feats_batch,  # voxel features
            "num_points": num_points,  # the number of points for each voxel
            "pc_voxel_id": pc_voxel_id,  # voxel id for each point
            "len_batch": len_batch
            #######
        }


class NuScenesDatasetSpconv(Dataset):
    """
    Dataset returning a lidar scene and associated labels.
    """

    def __init__(self, phase, config, transforms=None, cached_nuscenes=None):
        self.phase = phase
        self.labels = self.phase != "test"
        self.transforms = transforms

        if config["dataset"] == "nuscenes":
            self.voxel_size = [0.1, 0.1, 0.2]  # nuScenes
            self.point_cloud_range = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], dtype=np.float32)  # nuScenes
            MAX_POINTS_PER_VOXEL = 10  # nuScenes
            MAX_NUMBER_OF_VOXELS = 60000  # nuScenes
            self._voxel_generator = PointToVoxel(
                vsize_xyz=self.voxel_size,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=4,
                max_num_points_per_voxel=MAX_POINTS_PER_VOXEL,
                max_num_voxels=MAX_NUMBER_OF_VOXELS
            )
        else:
            raise Exception("Dataset unknown")
        # self.voxel_size = config["voxel_size"]
        # self.cylinder = config["cylindrical_coordinates"]
        self.config = config
        self.num_point_features = 4
        if 'data_root' in config:
            data_root = config['data_root']
        else:
            data_root = "datasets/nuscenes"

        self.panoptic = config['panoptic'] if 'panoptic' in config else False

        if phase != "test":
            self.nusc = cached_nuscenes if cached_nuscenes is not None else NuScenes(
                version="v1.0-trainval", dataroot=data_root, verbose=False
            )
        else:
            self.nusc = NuScenes(
                version="v1.0-test", dataroot=data_root, verbose=False
            )

        self.list_tokens = []

        # a skip ratio can be used to reduce the dataset size
        # and accelerate experiments
        if phase in ("val", "verifying"):
            skip_ratio = 1
        else:
            try:
                skip_ratio = config["dataset_skip_step"]
            except KeyError:
                skip_ratio = 1
        skip_counter = 0
        if phase in ("train", "val", "test"):
            phase_scenes = create_splits_scenes()[phase]
        elif phase == "parametrizing":
            phase_scenes = list(
                set(create_splits_scenes()["train"]) - set(CUSTOM_SPLIT)
            )
        elif phase == "verifying":
            phase_scenes = CUSTOM_SPLIT
        # create a list of all keyframe scenes
        for scene_idx in range(len(self.nusc.scene)):
            scene = self.nusc.scene[scene_idx]
            if scene["name"] in phase_scenes:
                skip_counter += 1
                if skip_counter % skip_ratio == 0:
                    self.create_list_of_tokens(scene)

        # labels' names lookup table
        self.eval_labels = {
            0: 0,
            1: 0,
            2: 7,  # person
            3: 7,  # person
            4: 7,  # person
            5: 0,
            6: 7,  # person
            7: 0,
            8: 0,
            9: 1,  # barrier
            10: 0,
            11: 0,
            12: 8,  #
            13: 0,
            14: 2,
            15: 3,
            16: 3,
            17: 4,
            18: 5,
            19: 0,
            20: 0,
            21: 6,
            22: 9,
            23: 10,
            24: 11,
            25: 12,
            26: 13,
            27: 14,
            28: 15,
            29: 0,
            30: 16,
            31: 0,
        }
        ########################

    def create_list_of_tokens(self, scene):
        # Get first in the scene
        current_sample_token = scene["first_sample_token"]

        # Loop to get all successive keyframes
        while current_sample_token != "":
            current_sample = self.nusc.get("sample", current_sample_token)
            next_sample_token = current_sample["next"]
            self.list_tokens.append(current_sample["data"]["LIDAR_TOP"])
            current_sample_token = next_sample_token

    def _voxelize(self, points):
        voxel_output = self._voxel_generator.generate_voxel_with_id(points)
        voxels, coordinates, num_points, indexes = voxel_output
        return voxels, coordinates, num_points, indexes

    def __len__(self):
        return len(self.list_tokens)

    def __getitem__(self, idx):
        lidar_token = self.list_tokens[idx]
        pointsensor = self.nusc.get("sample_data", lidar_token)
        pcl_path = os.path.join(self.nusc.dataroot, pointsensor["filename"])
        points = LidarPointCloud.from_file(pcl_path).points.T
        # get the points (4th coordinate is the point intensity)
        intensity = torch.tensor(points[:, 3:])
        pc = points[:, :3]
        if self.labels:
            lidarseg_labels_filename = os.path.join(
                self.nusc.dataroot, self.nusc.get("lidarseg", lidar_token)["filename"]
            )
            points_labels = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)

        pc = torch.tensor(pc)
        # apply the transforms (augmentation)
        if self.transforms:
            pc = self.transforms(pc)

        pc = torch.cat((pc, intensity), 1)
        voxels, coordinates, num_points, pc_voxel_id = self._voxelize(pc)
        # voxels: 每个voxel的特征, 包括 xyz+intensity
        # coordinates: 每个voxel的坐标, xyz
        # num_points:  每个voxel所包含的点数量
        # pc_voxel_id:  每个point所对应的voxel id
        discrete_coords = torch.cat(
            (
                torch.zeros(coordinates.shape[0], 1, dtype=torch.int32),
                coordinates,
            ),
            1,
        )

        if self.labels:
            points_labels = torch.tensor(
                np.vectorize(self.eval_labels.__getitem__)(points_labels),
                dtype=torch.int32,
            )
            unique_labels = points_labels[pc_voxel_id]  # point -> voxel  0~16
        # voxels: (15520,10,4)
        # 取平均
        # voxels_feats = torch.sum(voxels, dim=1)
        # voxels_feats = torch.div(voxels_feats, num_points.unsqueeze(-1))

        if self.labels:
            return (
                pc,  # point features
                discrete_coords,  # voxel的离散坐标,
                voxels,  # voxel features. shape: (num_voxels, max_num_points, 4) 比如 (15520, 10, 4)
                # 后续在 collate_fn 里面会对该特征做进一步的处理。
                unique_labels,  # voxel labels (sem)
                points_labels,  # point labels (sem)
                pc_voxel_id,  # voxel id of each point
                num_points,  # the number of points for each voxel
                # lidar_token,
            )
        else:
            return pc, discrete_coords, voxels, pc_voxel_id, num_points


def make_data_loader_nuscenes_spconv(config, phase, num_threads=0):
    """
    Create the data loader for a given phase and a number of threads.
    This function is not used with pytorch lightning, but is used when evaluating.
    """
    # select the desired transformations
    if phase == "train":
        transforms = make_transforms_clouds(config)
    else:
        transforms = None

    # instantiate the dataset
    dset = NuScenesDatasetSpconv(phase=phase, transforms=transforms, config=config)
    collate_fn = spconv_custom_collate_fn
    batch_size = config["batch_size"] // config["num_gpus"]

    # create the loader
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=phase == "train",
        num_workers=num_threads,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=phase == "train",
        worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
    )
    return loader
