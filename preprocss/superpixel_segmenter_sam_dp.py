import multiprocessing
import os
import argparse
import torch.multiprocessing

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from skimage.segmentation import slic
from nuscenes.nuscenes import NuScenes

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

sam = sam_model_registry["vit_h"](checkpoint="/home/chm/Codes/checkpoints/sam/sam_vit_h_4b8939.pth").cuda()
# sam = sam_model_registry["vit_h"](checkpoint="/home/njh/chm/chm/Codes/checkpoints/sam_vit_h_4b8939.pth")
# sam = sam_model_registry["vit_h"](checkpoint="/media/test/chm/Codes/checkpoints/sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)
print("---------")


def compute_slic(cam_token, nusc, mask_generator):
    print("1")
    cam = nusc.get("sample_data", cam_token)
    im_array = np.array(Image.open(os.path.join(nusc.dataroot, cam["filename"])))  # HxWxC
    masks = mask_generator.generate(im_array)

    #######
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)  # reverse=True, 由大到小
    sam_output = np.ones(im_array.shape[0:2], dtype=np.uint8) * 0
    segment_label = 1
    for ann in sorted_anns:
        m = ann['segmentation']  # (900,1600)
        sam_output[m] = segment_label
        segment_label += 1

        if len(sorted_anns) > 256:
            segment_label = 1

    im = Image.fromarray(sam_output)
    im.save("./superpixels/nuscenes/superpixels_sam_areaB2S/" + cam["token"] + ".png")
    ###################
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=False)  # reverse=True, 由大到小
    sam_output = np.ones(im_array.shape[0:2], dtype=np.uint8) * 0
    segment_label = 1
    for ann in sorted_anns:
        m = ann['segmentation']  # (900,1600)
        sam_output[m] = segment_label
        segment_label += 1

        if len(sorted_anns) > 256:
            segment_label = 1

    im = Image.fromarray(sam_output)
    im.save("./superpixels/nuscenes/superpixels_sam_areaS2B/" + cam["token"] + ".png")

    # img = np.ones((m.shape[0], m.shape[1], 3))
    # color_mask = np.random.random((1, 3)).tolist()[0]
    # for i in range(3):
    #     img[:, :, i] = color_mask[i]  # (900,1600,3)
    # ax.imshow(np.dstack((img, m * 0.35)))


def compute_slic_30(cam_token):
    cam = nusc.get("sample_data", cam_token)
    im = Image.open(os.path.join(nusc.dataroot, cam["filename"]))
    segments_slic = slic(
        im, n_segments=30, compactness=6, sigma=3.0, start_label=0
    ).astype(np.uint8)  #
    im = Image.fromarray(segments_slic)
    im.save(
        "./superpixels/nuscenes/superpixels_slic_30/" + cam["token"] + ".png"
    )


import torch.multiprocessing as mp

if __name__ == "__main__":
    nuscenes_path = "/home/chm/Datasets/nuScenes"
    # nuscenes_path = "/home/njh/chm/chm/Datasets/nuScenes/nuScenes"
    # nuscenes_path = "/media/test/stf/datasets/nuScenes"
    # nuscenes_path = "datasets/nuscenes"
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--model", type=str, default="minkunet", help="specify the model targeted, either minkunet or voxelnet"
    )
    assert os.path.exists(nuscenes_path), f"nuScenes not found in {nuscenes_path}"
    args = parser.parse_args()
    assert args.model in ["minkunet", "voxelnet"]
    nusc = NuScenes(
        version="v1.0-trainval", dataroot=nuscenes_path, verbose=False
    )
    os.makedirs("superpixels/nuscenes/superpixels_slic/", exist_ok=True)
    os.makedirs("superpixels/nuscenes/superpixels_sam_areaB2S/", exist_ok=True)
    os.makedirs("superpixels/nuscenes/superpixels_sam_areaS2B/", exist_ok=True)
    camera_list = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_FRONT_LEFT",
    ]

    # for scene_idx in tqdm(range(len(nusc.scene))):
    #     scene = nusc.scene[scene_idx]
    #     current_sample_token = scene["first_sample_token"]
    #     while current_sample_token != "":
    #         current_sample = nusc.get("sample", current_sample_token)
    #         if args.model == "minkunet":
    #             func = compute_slic
    #         elif args.model == "voxelnet":
    #             func = compute_slic_30
    #
    #         for camera_name in camera_list:
    #             func(current_sample["data"][camera_name])
    #
    #         # func(current_sample["data"][camera_list[0]])
    #         # p.map(
    #         #     func,
    #         #     [
    #         #         current_sample["data"][camera_name]
    #         #         for camera_name in camera_list
    #         #     ],
    #         # )
    #         current_sample_token = current_sample["next"]

    # 多线程

    ctx = multiprocessing.get_context("spawn")
    print(torch.multiprocessing.cpu_count())
    # pool = ctx.Pool(2)

    with ctx.Pool(2) as p:
        for scene_idx in tqdm(range(len(nusc.scene))):
            scene = nusc.scene[scene_idx]
            current_sample_token = scene["first_sample_token"]
            while current_sample_token != "":
                current_sample = nusc.get("sample", current_sample_token)
                if args.model == "minkunet":
                    func = compute_slic
                elif args.model == "voxelnet":
                    func = compute_slic_30
                # mp.spawn(func, nprocs=2, args=(args, cfg))
                # for camera_name in camera_list:
                #     p.apply_async(func, args=(current_sample["data"][camera_name], nusc, mask_generator))

                p.map(
                    func,
                    [
                        (current_sample["data"][camera_name], nusc, mask_generator)
                        for camera_name in camera_list
                    ],
                )
                current_sample_token = current_sample["next"]
