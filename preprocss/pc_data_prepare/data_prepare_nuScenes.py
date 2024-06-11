import argparse
import os

import numpy as np
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm

from preprocss.pc_data_prepare.helper_ply import write_ply


def load_pc_nuscenes(pc_path):
    """
        Loads LIDAR data from binary numpy format. Data is stored as (x, y, z, intensity, ring index).
        :param file_name: Path of the pointcloud file on disk.
        :return: LidarPointCloud instance (x, y, z, intensity).
    """
    scan = np.fromfile(pc_path, dtype=np.float32)
    points = scan.reshape((-1, 5))[:, :4]  # (x, y, z, intensity, ring index) -> (x, y, z, intensity)

    return points


def load_label_nuscenes(label_path):
    remap_labels = {
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
    point_labels = np.fromfile(label_path, dtype=np.uint8)
    point_labels = np.vectorize(remap_labels.__getitem__)(point_labels)

    sem_label = point_labels

    return sem_label.astype(np.int32), [0, 0, 0]


def load_pc_kitti(pc_path):
    scan = np.fromfile(pc_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    # points = scan[:, 0:3]  # get xyz
    # return points
    return scan


def load_label_kitti(label_path, remap_lut):
    label = np.fromfile(label_path, dtype=np.uint32)
    label = label.reshape((-1))
    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half
    assert ((sem_label + (inst_label << 16) == label).all())
    sem_colors = [DATA["color_map"][i] for i in sem_label]
    sem_colors = np.array(sem_colors)
    sem_label = remap_lut[sem_label]
    return sem_label.astype(np.int32), sem_colors.astype(np.uint8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/user/SSD2/dataset/sequences', help='raw data path')
    parser.add_argument('--processed_data_path', type=str, default='data/SemanticKITTI/dataset/sequences')
    args = parser.parse_args()

    out_dir = '/home/chm/workspace/datasets/nuScenes/ply_data'
    nuscenes_path = "/home/chm/workspace/datasets/nuScenes"  # 241
    nusc = NuScenes(
        version="v1.0-trainval", dataroot=nuscenes_path, verbose=True
    )
    for scene_idx in tqdm(range(len(nusc.scene))):
        ##
        print('scene_idx ' + str(scene_idx) + ' start')
        scene = nusc.scene[scene_idx]
        current_sample_token = scene["first_sample_token"]
        while current_sample_token != "":
            current_sample = nusc.get("sample", current_sample_token)
            ##### 处理每个sample的逻辑
            lidar_token = current_sample['data']['LIDAR_TOP']
            lidar = nusc.get("sample_data", lidar_token)
            lidar_path = os.path.join(nusc.dataroot, lidar['filename'])

            points = load_pc_nuscenes(lidar_path)

            lidar_label_path = os.path.join(
                nusc.dataroot, nusc.get("lidarseg", lidar_token)['filename']
            )

            labels, sem_colors = load_label_nuscenes(lidar_label_path)

            write_ply(os.path.join(out_dir, lidar_token) + '.ply',
                      [points[:, 0:3], labels, points[:, 3]],
                      ['x', 'y', 'z', 'class', 'remission']
                      )
            #####
            current_sample_token = current_sample["next"]
#
# for seq_id in seq_list:
#     print('sequence ' + seq_id + ' start')
#     seq_path = join(args.data_path, seq_id)
#     seq_path_out = join(args.processed_data_path, seq_id)
#     pc_path = join(seq_path, 'velodyne')
#     os.makedirs(seq_path_out) if not exists(seq_path_out) else None
#
#     if int(seq_id) < 11:
#         label_path = join(seq_path, 'labels')
#         scan_list = np.sort(os.listdir(pc_path))
#         for scan_id in scan_list:
#             print(seq_id, scan_id)
#             points = load_pc_kitti(join(pc_path, scan_id))
#             labels, sem_colors = load_label_kitti(join(label_path, str(scan_id[:-4]) + '.label'), remap_lut)
#             labels = labels.squeeze()[:, None]
#             write_ply(join(seq_path_out, scan_id)[:-4] + '.ply', [points[:, 0:3], sem_colors, labels, points[:, 3]],
#                       ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'remission'])
#
#     else:
#         scan_list = np.sort(os.listdir(pc_path))
#         for scan_id in scan_list:
#             print(seq_id, scan_id)
#             points = load_pc_kitti(join(pc_path, scan_id))
#             labels = -np.ones_like(points)[:, 0]
#             labels = labels.squeeze()[:, None]
#             sem_colors = -np.ones((labels.shape[0], 3)).astype((np.uint8))
#             write_ply(join(seq_path_out, scan_id)[:-4] + '.ply', [points[:, 0:3], sem_colors, labels, points[:, 3]],
#                       ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'remission'])
