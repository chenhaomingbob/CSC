import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.segmentation import slic
from nuscenes.nuscenes import NuScenes
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def compute_sam(cam_token, save_dir):
    cam = nusc.get("sample_data", cam_token)
    image = cv2.imread(os.path.join(nusc.dataroot, cam["filename"]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # save_dir = "../superpixels/nuscenes/superpixels_sam_areaB2S_v2/"
    img_save_path = save_dir + cam["token"] + ".png"
    # img_name = save_dir + cam["token"] + ".png"
    if os.path.exists(img_save_path):
        return

    masks = mask_generator.generate(image)
    #######
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)  # reverse=True,
    sam_output = np.zeros(image.shape[0:2])
    for id, ann in enumerate(sorted_anns):
        m = ann['segmentation']  # (900,1600)
        sam_output[m] = id

    im = Image.fromarray(sam_output.astype(np.uint8))
    im.save(img_save_path)
    ###################
    # sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=False)  # reverse=True, 由大到小
    # sam_output = np.ones(im_array.shape[0:2], dtype=np.uint8) * 0
    # segment_label = 1
    # for ann in sorted_anns:
    #     m = ann['segmentation']  # (900,1600)
    #     sam_output[m] = segment_label
    #     segment_label += 1
    #
    #     if len(sorted_anns) > 256:
    #         segment_label = 1
    #
    # im = Image.fromarray(sam_output)
    # im.save("./superpixels/nuscenes/superpixels_sam_areaS2B_v2/" + cam["token"] + ".png")

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


if __name__ == "__main__":
    # nuscenes_path = "/home/chm/workspace/datasets/nuScenes"
    # nuscenes_path = "/home/chm/Datasets/nuScenes"
    # nuscenes_path = "/home/njh/chm/chm/Datasets/nuScenes/nuScenes"
    nuscenes_path = "/media/test/stf/datasets/nuScenes"
    # nuscenes_path = "datasets/nuscenes"
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--model", type=str, default="minkunet", help="specify the model targeted, either minkunet or voxelnet"
    )
    parser.add_argument("--split", type=int, default=2)
    parser.add_argument("--id", type=int, default=-1)
    assert os.path.exists(nuscenes_path), f"nuScenes not found in {nuscenes_path}"
    args = parser.parse_args()
    assert args.model in ["minkunet", "voxelnet"]
    nusc = NuScenes(
        version="v1.0-trainval", dataroot=nuscenes_path, verbose=True
    )

    save_dir = "../superpixels/nuscenes/superpixels_sam_areaB2S_v2/"
    os.makedirs(save_dir, exist_ok=True)
    camera_list = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_FRONT_LEFT",
    ]

    sam = sam_model_registry["vit_h"](checkpoint="/home/test/media_chm/Codes/3D_SLidR/superpixels/nuscenes/sam_vit_h_4b8939.pth").cuda()
    mask_generator = SamAutomaticMaskGenerator(sam)
    print(f"-----------args.split={args.split}-args.id={args.id}-------------")

    for scene_idx in tqdm(range(len(nusc.scene))):
        if args.id == -1 or scene_idx % args.split == args.id:
            print(scene_idx)
            scene = nusc.scene[scene_idx]
            current_sample_token = scene["first_sample_token"]
            while current_sample_token != "":
                current_sample = nusc.get("sample", current_sample_token)
                if args.model == "minkunet":
                    func = compute_sam
                elif args.model == "voxelnet":
                    func = compute_slic_30

                for camera_name in camera_list:
                    func(current_sample["data"][camera_name], save_dir)

                current_sample_token = current_sample["next"]
