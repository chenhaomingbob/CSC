import os.path

import torch
import shutil
import argparse

import yaml

from downstream.evaluate import evaluate
from utils.read_config import generate_config
from downstream.semantic_segmentation.dataloader_builder import make_dataloader as make_semseg_dataloader
from downstream.semantic_segmentation.model_builder import make_model as make_semseg_model


def main():
    """
    Code for launching the downstream evaluation
    """
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        '--backbone', default='minkunet', choices=['minkunet', 'voxelnet', 'cylinder3d'],
        help="Select a 3d backbone in ['minkunet','voxelnet','cylinder3d']."
             "\nNote:"
             "\nminkunet: semantic segmentation."
             "\nvoxelnet: object detection."
             "\ncylinder3d: panoptic segmentation."
    )
    parser.add_argument(
        "--dataset", default='nuScenes', choices=['nuScenes', 'SemanticKITTI'],
        help="Choose a dataset, including nuScenes and KITTI"
    )
    parser.add_argument(
        "--task", default='semseg', choices=['semseg', 'objdet', 'panseg'],
        help="Select a downstream task in ['semseg','panseg','objdet']."
             "\nsemseg: semantic segmentation."
             "\nobjdet: object detection."
             "\npanseg: panoptic segmentation."
    )
    parser.add_argument(
        "--data_split", default='validation', choices=['train', 'val', 'test'],
        help="Select the split of dataset in ['training','validation','test']."
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Select the weights of the fine-tuned 3d network"
    )
    parser.add_argument(
        "--working_base_dir", type=str, default="./output/evaluate",
        help="The path of saving configs & results"
    )
    parser.add_argument(
        "--exp_name", type=str, default="cp",
    )
    args = parser.parse_args()
    ######################################################
    ##      Select config depend on task & dataset      ##
    ######################################################
    args.dataset = args.dataset.lower()
    args.base_cfg_file = os.path.join("./config/evaluate", f"{args.dataset}_{args.task}.yaml")
    args.working_dir = \
        os.path.join(args.working_base_dir,
                     f"{args.dataset}_{args.task}_{args.backbone}_{args.data_split}",
                     args.exp_name)
    os.makedirs(args.working_dir, exist_ok=True)
    if not os.path.exists(args.base_cfg_file):
        raise Exception(f"The config file is not exist. The path of config is: {args.base_cfg_file}")
    if not os.path.exists(args.weights):
        raise Exception(f"The weights of 3d model is not exist. The path of weights is: {args.finetuned_weights}")
    config = generate_config(args.base_cfg_file)
    config['weights'] = args.weights
    config['data_split'] = args.data_split
    config['task'] = args.task
    config['dataset'] = args.dataset
    config['backbone'] = args.backbone
    config['working_dir'] = args.working_dir
    config['exp_name'] = args.exp_name
    ######################################################
    # print config
    print("\n" + "\n".join(list(map(lambda x: f"{x[0]:20}: {x[1]}", config.items()))))
    print("Creating the loaders")
    # save config file
    config_save_path = os.path.join(args.working_dir, 'config_file.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    ######################################################
    dataloader, model, eval_func = None, None, None
    if args.task == 'semseg':
        from downstream.semantic_segmentation.evaluate_semseg import evaluate_semseg
        print("Creating the dataloader for semantic segmentation")
        dataloader = make_semseg_dataloader(config)
        print("Creating the model for semantic segmentation")
        model = make_semseg_model(config, load_path=args.weights).cuda()
        eval_func = evaluate_semseg
    elif args.task == 'objdet':
        assert Exception("Please refer to https://github.com/zaiweizhang/OpenPCDet "
                         "or https://github.com/open-mmlab/OpenPCDet")

    elif args.task == 'panseg':
        from downstream.panoptic_segmentation.dataloader_kitti import make_data_loader_semkitti
        from downstream.panoptic_segmentation.dataloader_nuscenes import make_data_loader_nuscenes

        assert Exception("Not finish the code")

    eval_func(model=model, dataloader=dataloader, config=config)


if __name__ == "__main__":
    main()
