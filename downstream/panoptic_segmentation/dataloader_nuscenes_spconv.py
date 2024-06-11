import os

import numpy as np
import torch
from MinkowskiEngine.utils import sparse_quantize
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.splits import create_splits_scenes
from torch.utils.data import Dataset
import numba as nb

from utils.transforms import make_transforms_clouds
from spconv.pytorch.utils import PointToVoxel
from downstream.panoptic_segmentation.process_panoptic import PanopticLabelGenerator
from downstream.panoptic_segmentation.dataset_utils import collate_dataset_info

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

    labelized = len(input) >= 8

    if labelized:
        # 8 items
        pc_features, grid_ind, pc_xyz, point_sem_label, point_inst_label, voxel_sem_label, \
        evaluation_point_sem_label, evaluation_point_inst_label, center, offset = input
    else:
        # 3 items
        pc_features, grid_ind, pc_xyz = input

    if labelized:
        pc_features = [torch.Tensor(data) for data in pc_features]
        grid_ind = [torch.Tensor(data) for data in grid_ind]

        voxel_sem_label = torch.tensor(np.stack(voxel_sem_label, axis=0)).long()
        center = torch.tensor(np.stack(center, axis=0))
        offset = torch.tensor(np.stack(offset, axis=0))

        return {
            "pc_features": pc_features,
            "pc_grid_index": grid_ind,
            "pc_xyz": pc_xyz,
            "point_sem_label": point_sem_label,
            "point_inst_label": point_inst_label,
            "evaluation_point_sem_label": evaluation_point_sem_label,
            "evaluation_point_inst_label": evaluation_point_inst_label,
            "voxel_sem_label": voxel_sem_label,  # # (B,480,360,32)
            "gt_center": center,  # (B,1,480,360)
            "gt_offset": offset,  # (B,2,480,360)
        }

    else:
        return {
            "pc_features": pc_features,
            "pc_grid_index": grid_ind,
            "pc_xyz": pc_xyz,
            # "point_sem_label": point_sem_label,
            # "point_inst_label": point_inst_label,
            # "voxel_sem_label": voxel_sem_label,
            # "gt_center": center,
            # "gt_offset": offset
        }


class NuScenesDatasetSpconv(Dataset):
    """
    Dataset returning a lidar scene and associated labels.
    """

    def __init__(self, phase, config, transforms=None, cached_nuscenes=None):
        self.phase = phase

        # self.labels = config['pre-training'] if 'pre-training' in config else True
        self.labels = self.phase != "test"
        self.transforms = transforms

        self.use_polar = config['use_polar'] if 'use_polar' in config else False
        dataset_name = config["dataset"]
        dataset_type = config['type'] if 'type' in config else 'full'
        assert self.use_polar
        if dataset_name == "nuscenes":
            if self.use_polar:
                # 极坐标系下
                self.point_cloud_range = np.array([0, -3.1415926, -3, 50, 3.1415926, 3], dtype=np.float32)
                self.grid_size = [480, 360, 32]
                # self.voxel_size: array([0.10416667, 0.01745329, 0.1875    ])
                self.voxel_size = (self.point_cloud_range[3:] - self.point_cloud_range[0:3]) / self.grid_size
                MAX_POINTS_PER_VOXEL = 10  # nuScenes
                MAX_NUMBER_OF_VOXELS = 100000  # nuScenes
                self._voxel_generator = PointToVoxel(
                    vsize_xyz=self.voxel_size,
                    coors_range_xyz=self.point_cloud_range,
                    num_point_features=4,
                    max_num_points_per_voxel=MAX_POINTS_PER_VOXEL,
                    max_num_voxels=MAX_NUMBER_OF_VOXELS
                )
            else:
                # 笛卡尔坐标系
                self.voxel_size = [0.1, 0.1, 0.2]  # nuScenes
                self.point_cloud_range = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], dtype=np.float32)  # nuScenes
                #
                MAX_POINTS_PER_VOXEL = 10  # nuScenes
                MAX_NUMBER_OF_VOXELS = 60000  # nuScenes
                self._voxel_generator = PointToVoxel(
                    vsize_xyz=self.voxel_size,
                    coors_range_xyz=self.point_cloud_range,
                    num_point_features=4,
                    max_num_points_per_voxel=MAX_POINTS_PER_VOXEL,
                    max_num_voxels=MAX_NUMBER_OF_VOXELS
                )
                self.grid_size = self._voxel_generator.grid_size
            self.max_bound = self.point_cloud_range[3:]
            self.min_bound = self.point_cloud_range[0:3]
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
        ########################
        self.unique_label, self.unique_label_str, self.thing_list = collate_dataset_info(dataset_name)  # 去掉了noise类
        self.denoise_thing_list = [i - 1 for i in self.thing_list]  # nuScenes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.panoptic = config['panoptic'] if 'panoptic' in config else False
        self.ignore_label = len(self.unique_label) + 1  # nuScenes: 17
        self.panoptic_label_generator = PanopticLabelGenerator(self.grid_size, sigma=5, polar=self.use_polar)
        ################
        if phase != "test":
            if dataset_type == 'full':
                self.nusc = cached_nuscenes if cached_nuscenes is not None else NuScenes(
                    version="v1.0-trainval", dataroot=data_root, verbose=False
                )
            else:
                self.nusc = cached_nuscenes if cached_nuscenes is not None else NuScenes(
                    version="v1.0-mini", dataroot=data_root, verbose=False
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
        # 0~16
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
        # return 100
        # if self.phase == 'val':
        #     return 10

        return len(self.list_tokens)

    def __getitem__(self, idx):
        lidar_token = self.list_tokens[idx]
        pointsensor = self.nusc.get("sample_data", lidar_token)
        pcl_path = os.path.join(self.nusc.dataroot, pointsensor["filename"])
        points = LidarPointCloud.from_file(pcl_path).points.T
        # get the points (4th coordinate is the point intensity)
        # intensity = torch.tensor(points[:, 3:])
        intensity = points[:, 3:]
        pc_xyz = points[:, :3]

        if self.labels:
            lidarseg_labels_filename = os.path.join(
                self.nusc.dataroot, self.nusc.get("lidarseg", lidar_token)["filename"]
            )
            panoptic_labels_filename = os.path.join(
                self.nusc.dataroot, self.nusc.get('panoptic', lidar_token)['filename']
            )
            point_inst_label = np.load(panoptic_labels_filename)['data']
            point_sem_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
            # 以nuScenes为例, [0~16], 0为noise label
            point_sem_label = np.vectorize(self.eval_labels.__getitem__)(point_sem_label)
            # 以nuScenes为例, [0~16], 0为noise label
            evaluation_point_sem_label = np.copy(point_sem_label)
            evaluation_point_inst_label = np.copy(point_inst_label)

            noise_mask = point_sem_label == 0
            point_sem_label[noise_mask] = self.ignore_label  # 以nuScenes为例, 0->17, [1~17]
            point_sem_label = np.subtract(point_sem_label, 1)  # 以nuScenes为例, [0~16]
            process_ignore_label = self.ignore_label - 1  #

        pc_xyz = torch.tensor(pc_xyz)
        if self.transforms:
            pc_xyz = self.transforms(pc_xyz)
        pc_xyz = np.array(pc_xyz)

        intensity = torch.tensor(intensity)
        grid_size = np.array(self.grid_size)
        pc_xyz_polar = cart2polar(pc_xyz)
        crop_range = self.max_bound - self.min_bound
        intervals = crop_range / (grid_size - 1)
        grid_ind = (np.floor((np.clip(pc_xyz_polar, self.min_bound, self.max_bound) - self.min_bound) / intervals)) \
            .astype(np.int64)

        # voxel_position = np.zeros(grid_size, dtype=np.float32)
        dim_array = np.ones(len(grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(grid_size) * intervals.reshape(dim_array) + self.min_bound.reshape(
            dim_array)  # 每个point的voxel位置

        pc_features = np.concatenate([pc_xyz_polar, intensity], axis=1)
        if self.labels:
            # process voxel labels
            processed_label = np.ones(grid_size, dtype=np.uint8) * 255  # [480,360,32]
            label_voxel_pair = np.concatenate([grid_ind, point_sem_label], axis=1)  # [N,4]
            label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])),
                               :]  # [N,4]
            processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)  # [480,360,32]
            processed_label[processed_label==255] = process_ignore_label
            # get thing points mask
            point_inst_label = point_inst_label.reshape(-1, 1)
            mask = np.zeros_like(point_sem_label, dtype=bool)
            for label in self.denoise_thing_list:
                mask[np.logical_and(point_sem_label == label, point_inst_label != 0)] = True
            # training/validation

            instance_label = point_inst_label[mask].squeeze()
            unique_label = np.unique(instance_label)
            unique_label_dict = {label: idx + 1 for idx, label in enumerate(unique_label)}
            if instance_label.size > 1:
                inst_label = np.vectorize(unique_label_dict.__getitem__)(instance_label)
                # process panoptic
                processed_inst = np.ones(grid_size[:2], dtype=np.uint8) * 255
                inst_voxel_pair = np.concatenate([grid_ind[mask[:, 0], :2], inst_label[..., np.newaxis]], axis=1)
                inst_voxel_pair = inst_voxel_pair[
                                  np.lexsort((grid_ind[mask[:, 0], 0], grid_ind[mask[:, 0], 1])), :]
                processed_inst = nb_process_inst(np.copy(processed_inst), inst_voxel_pair)
            else:
                processed_inst = None

            center, center_points, offset = self.panoptic_label_generator(
                point_inst_label[mask],
                pc_xyz[mask[:, 0]],
                processed_inst,
                voxel_position[:2, :, :, 0],
                unique_label_dict, self.min_bound, intervals)

            voxel_sem_label = processed_label
            return (
                pc_features,  # 点的特征（xyz+intensity /  xyz_polar + intensity），将会作为网络的输入
                grid_ind,  # 每个point的位置信息
                pc_xyz,  # 点云在笛卡尔坐标系下的位置。
                # point_sem_label: np.ndarray, point sem labels for training.
                # range from 0 to 16, where 0 is barrier class
                point_sem_label,
                # point_inst_label: np.ndarray, point inst labels for training.
                point_inst_label,
                # voxel_sem_label: np.ndarray, voxel sem labels for training.
                # range from 0 to 16, where 0 is barrier class
                voxel_sem_label,
                # evaluation_point_sem_label: np.ndarray, point-wise semantic labels for evaluation.
                # range from 0 to 15, where 0 is noise class
                evaluation_point_sem_label,
                # evaluation_point_inst_label: np.ndarray, point-wise instance labels for evaluation.
                evaluation_point_inst_label,
                # center: gt_center
                center,
                # offset: gt_offset
                offset,
            )

        else:
            return (
                pc_features,
                grid_ind,
                pc_xyz
            )

            # # test?
            # pc = np.concatenate([pc_xyz_polar, intensity], axis=1)
            # voxels, coordinates, num_points, pc_voxel_id = self._voxelize(pc)
            # discrete_coords = grid_ind
            #
            # return pc, discrete_coords, voxels, pc_voxel_id, num_points


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


# @nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', cache=True, parallel=True, nopython=True)
@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', cache=True, parallel=False, nopython=True)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label


# boolean	b1
# bool_	b1
# byte	u1
# uint8	u1
# uint16	u2
# uint32	u4
# uint64	u8
# char	i1
# int8	i1
# int16	i2
# int32	i4
# int64	i8
# float_	f4
# float32	f4
# double	f8
# float64	f8
# complex64	c8
# complex128	c16
# @nb.jit('u1[:,:](u1[:,:],i8[:,:])', cache=True, parallel=False, nopython=True)
@nb.jit('u1[:,:](u1[:,:],i8[:,:])', cache=True, parallel=False, nopython=True)
def nb_process_inst(processed_inst, sorted_inst_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_inst_voxel_pair[0, 2]] = 1
    cur_sear_ind = sorted_inst_voxel_pair[0, :2]
    for i in range(1, sorted_inst_voxel_pair.shape[0]):
        cur_ind = sorted_inst_voxel_pair[i, :2]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_inst[cur_sear_ind[0], cur_sear_ind[1]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_inst_voxel_pair[i, 2]] += 1
    processed_inst[cur_sear_ind[0], cur_sear_ind[1]] = np.argmax(counter)
    return processed_inst


def cart2polar(input_xyz_cart):
    rho = np.sqrt(input_xyz_cart[:, 0] ** 2 + input_xyz_cart[:, 1] ** 2)
    phi = np.arctan2(input_xyz_cart[:, 1], input_xyz_cart[:, 0])
    return np.stack((rho, phi, input_xyz_cart[:, 2]), axis=1)


def polar2cat(input_xyz_polar):
    # print(input_xyz_polar.shape)
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)
