import copy
import os

import MinkowskiEngine as ME
import numpy as np
import torch
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion
from torch.utils.data import Dataset


def minkunet_collate_pair_fn(list_data):
    """
    Collate function adapted for creating batches with MinkowskiEngine.
    """
    (
        coords,
        feats,
        images,
        pairing_points,
        pairing_images,
        inverse_indexes,
        superpixels,
        image_path_list
    ) = list(zip(*list_data))
    batch_n_points, batch_n_pairings = [], []

    offset = 0
    for batch_id in range(len(coords)):
        # Move batchids to the beginning
        coords[batch_id][:, 0] = batch_id
        pairing_points[batch_id][:] += offset
        pairing_images[batch_id][:, 0] += batch_id * images[0].shape[0]

        batch_n_points.append(coords[batch_id].shape[0])
        batch_n_pairings.append(pairing_points[batch_id].shape[0])
        offset += coords[batch_id].shape[0]

    # Concatenate all lists
    coords_batch = torch.cat(coords, 0).int()  # 4个sample的坐标
    pairing_points = torch.tensor(np.concatenate(pairing_points))  #
    pairing_images = torch.tensor(np.concatenate(pairing_images))
    feats_batch = torch.cat(feats, 0).float()
    images_batch = torch.cat(images, 0).float()
    superpixels_batch = torch.tensor(np.concatenate(superpixels))

    return {
        "sinput_C": coords_batch,  # 坐标
        "sinput_F": feats_batch,  # 特征
        "input_I": images_batch,  # 图像batch
        "pairing_points": pairing_points,  #
        "pairing_images": pairing_images,
        "batch_n_pairings": batch_n_pairings,
        "inverse_indexes": inverse_indexes,
        "superpixels": superpixels_batch,

        ######
        "image_path": image_path_list

    }


class NuScenesMatchDataset(Dataset):
    """
    Dataset matching a 3D points cloud and an image using projection.
    """

    def __init__(
            self,
            phase,
            config,
            shuffle=False,
            cloud_transforms=None,
            mixed_transforms=None,
            **kwargs,
    ):
        self.phase = phase
        self.shuffle = shuffle
        self.cloud_transforms = cloud_transforms
        self.mixed_transforms = mixed_transforms
        self.voxel_size = config["voxel_size"]
        self.cylinder = config["cylindrical_coordinates"]
        self.superpixels_type = config["superpixels_type"]
        self.bilinear_decoder = config["decoder"] == "bilinear"

        if "cached_nuscenes" in kwargs:
            self.nusc = kwargs["cached_nuscenes"]
        else:
            self.nusc = NuScenes(
                version="v1.0-trainval", dataroot="datasets/nuscenes", verbose=False
            )

        self.list_keyframes = []
        # a skip ratio can be used to reduce the dataset size and accelerate experiments
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
        # create a list of camera & lidar scans
        for scene_idx in range(len(self.nusc.scene)):
            scene = self.nusc.scene[scene_idx]
            if scene["name"] in phase_scenes:
                skip_counter += 1
                if skip_counter % skip_ratio == 0:
                    self.create_list_of_scans(scene)

    def create_list_of_scans(self, scene):
        # Get first and last keyframe in the scene
        current_sample_token = scene["first_sample_token"]

        # Loop to get all successive keyframes
        list_data = []
        while current_sample_token != "":
            current_sample = self.nusc.get("sample", current_sample_token)
            list_data.append(current_sample["data"])
            current_sample_token = current_sample["next"]

        # Add new scans in the list
        self.list_keyframes.extend(list_data)

    def map_pointcloud_to_image(self, data, min_dist: float = 1.0):
        """
        Given a lidar token and camera sample_data token, load pointcloud and map it to
        the image plane. Code adapted from nuscenes-devkit
        https://github.com/nutonomy/nuscenes-devkit.
        :param min_dist: Distance from the camera below which points are discarded.
        """
        pointsensor = self.nusc.get("sample_data", data["LIDAR_TOP"])
        pcl_path = os.path.join(self.nusc.dataroot, pointsensor["filename"])
        pc_original = LidarPointCloud.from_file(pcl_path)
        pc_ref = pc_original.points

        images = []
        superpixels = []
        img_path_list = []
        pairing_points = np.empty(0, dtype=np.int64)
        pairing_images = np.empty((0, 3), dtype=np.int64)
        camera_list = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_FRONT_LEFT",
        ]
        if self.shuffle:
            np.random.shuffle(camera_list)
        for i, camera_name in enumerate(camera_list):
            pc = copy.deepcopy(pc_original)
            cam = self.nusc.get("sample_data", data[camera_name])
            img_path = os.path.join(self.nusc.dataroot, cam["filename"])
            img_path_list.append(img_path)
            im = np.array(Image.open(img_path))
            sp = Image.open(
                f"superpixels/nuscenes/"
                f"superpixels_{self.superpixels_type}/{cam['token']}.png"
            )
            superpixels.append(np.array(sp))  # 超像素

            # Points live in the point sensor frame. So they need to be transformed via
            # global to the image plane.
            # First step: transform the pointcloud to the ego vehicle frame for the
            # timestamp of the sweep.
            cs_record = self.nusc.get(
                "calibrated_sensor", pointsensor["calibrated_sensor_token"]
            )
            pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
            pc.translate(np.array(cs_record["translation"]))

            # Second step: transform from ego to the global frame.
            poserecord = self.nusc.get("ego_pose", pointsensor["ego_pose_token"])
            pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
            pc.translate(np.array(poserecord["translation"]))

            # Third step: transform from global into the ego vehicle frame for the
            # timestamp of the image.
            poserecord = self.nusc.get("ego_pose", cam["ego_pose_token"])
            pc.translate(-np.array(poserecord["translation"]))
            pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

            # Fourth step: transform from ego into the camera.
            cs_record = self.nusc.get(
                "calibrated_sensor", cam["calibrated_sensor_token"]
            )
            pc.translate(-np.array(cs_record["translation"]))
            pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)

            # Fifth step: actually take a "picture" of the point cloud.
            # Grab the depths (camera frame z axis points away from the camera).
            depths = pc.points[2, :]

            # Take the actual picture
            # (matrix multiplication with camera-matrix + renormalization).
            points = view_points(
                pc.points[:3, :],
                np.array(cs_record["camera_intrinsic"]),
                normalize=True,
            )

            # Remove points that are either outside or behind the camera.
            # Also make sure points are at least 1m in front of the camera to avoid
            # seeing the lidar points on the camera
            # casing for non-keyframes which are slightly out of sync.
            points = points[:2].T
            mask = np.ones(depths.shape[0], dtype=bool)  #
            mask = np.logical_and(mask, depths > min_dist)  #
            mask = np.logical_and(mask, points[:, 0] > 0)
            mask = np.logical_and(mask, points[:, 0] < im.shape[1] - 1)  # height
            mask = np.logical_and(mask, points[:, 1] > 0)
            mask = np.logical_and(mask, points[:, 1] < im.shape[0] - 1)  # width
            matching_points = np.where(mask)[0]  # 匹配点的索引
            matching_pixels = np.round(
                np.flip(points[matching_points], axis=1)
            ).astype(np.int64)  # [height, width] -> [width, height]  图像上对应的像素
            images.append(im / 255)
            pairing_points = np.concatenate((pairing_points, matching_points))
            pairing_images = np.concatenate(
                (
                    pairing_images,
                    np.concatenate(
                        (
                            np.ones((matching_pixels.shape[0], 1), dtype=np.int64) * i,
                            matching_pixels,
                        ),
                        axis=1,
                    ),
                )
            )
        #   pc_ref.T:
        #   images: List
        #   pairing_points:
        #   pairing_images:
        #   superpixels:
        return pc_ref.T, images, pairing_points, pairing_images, np.stack(superpixels), img_path_list

    def __len__(self):
        return len(self.list_keyframes)

    def __getitem__(self, idx):
        (
            pc,
            images,
            pairing_points,  # 与image相匹配的point
            pairing_images,  # 与pc相匹配的pixel
            superpixels,  # 超像素
            img_path_list
        ) = self.map_pointcloud_to_image(self.list_keyframes[idx])
        superpixels = torch.tensor(superpixels)

        intensity = torch.tensor(pc[:, 3:])  # (34688) vs (22215)
        pc = torch.tensor(pc[:, :3])
        images = torch.tensor(np.array(images, dtype=np.float32).transpose(0, 3, 1, 2))

        # 点云数据增强
        if self.cloud_transforms:
            pc = self.cloud_transforms(pc)
        #
        if self.mixed_transforms:
            (
                pc,
                intensity,
                images,
                pairing_points,
                pairing_images,
                superpixels,
            ) = self.mixed_transforms(pc, intensity, images, pairing_points, pairing_images, superpixels)

        if self.cylinder:
            # Transform to cylinder coordinate and scale for voxel size
            x, y, z = pc.T
            rho = torch.sqrt(x ** 2 + y ** 2) / self.voxel_size
            phi = torch.atan2(y, x) * 180 / np.pi  # corresponds to a split each 1°
            z = z / self.voxel_size
            coords_aug = torch.cat((rho[:, None], phi[:, None], z[:, None]), 1)
        else:
            coords_aug = pc / self.voxel_size

        # Voxelization with MinkowskiEngine
        discrete_coords, indexes, inverse_indexes = ME.utils.sparse_quantize(
            coords_aug.contiguous(), return_index=True, return_inverse=True
        )  # voxel
        # indexes here are the indexes of points kept after the voxelization
        pairing_points = inverse_indexes[pairing_points]  # voxel
        unique_feats = intensity[indexes]  #

        discrete_coords = torch.cat(
            (
                torch.zeros(discrete_coords.shape[0], 1, dtype=torch.int32),
                discrete_coords,
            ),
            dim=1,
        )

        return (
            discrete_coords,  # voxel 坐标 (N,1)
            unique_feats,  # voxel 特征   (N,1)
            images,
            pairing_points,  # 存在映射的voxel坐标  8136: 全部的非空voxel数量 13235
            pairing_images,  # 存在映射的pixel坐标
            # point与image的关系, 可能是many-to-one
            inverse_indexes,  # voxel -> point
            superpixels,  # 图像的超像素
            img_path_list
        )
