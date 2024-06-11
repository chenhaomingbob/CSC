import argparse
import gc
import os
import torch
import torch.nn as nn

torch.set_float32_matmul_precision('high')
import MinkowskiEngine as ME

########### lightning v2.x #################
import pytorch_lightning

if pytorch_lightning.__version__.startswith("1"):
    pl_version = 1
    min_version = pytorch_lightning.__version__.split('.')[1]
    import pytorch_lightning as pl
    from pytorch_lightning.utilities import rank_zero_only
    from pytorch_lightning.plugins import DDPPlugin
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
elif pytorch_lightning.__version__.startswith("2"):
    pl_version = 2
    import lightning.pytorch as pl
    from lightning.pytorch.utilities import rank_zero_only
    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from downstream.panoptic_segmentation.lightning_datamodule import DownstreamDataModule
from downstream.panoptic_segmentation.lightning_trainer import LightningDownstream
from downstream.panoptic_segmentation.lightning_trainer_spconv import LightningDownstreamSpconv
from downstream.panoptic_segmentation.lightning_trainer_cylinder3d import LightningDownstreamCylinder3D
from downstream.panoptic_segmentation.model_builder_panoptic import make_model
from utils.common_utils import create_logger
from utils.read_config import generate_config


def main():
    """
    Code for launching the downstream training
    """
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file", type=str, default="config/semseg_nuscenes.yaml", help="specify the config for training"
    )
    parser.add_argument(
        "--resume_path", type=str, default=None, help="provide a path to resume an incomplete training"
    )
    parser.add_argument(
        "--pretraining_path", type=str, default=None, help="provide a path to pre-trained weights"
    )
    parser.add_argument(
        "--exp_name", type=str, default='default', help="provide a path to pre-trained weights"
    )
    args = parser.parse_args()
    config = generate_config(args.cfg_file)
    if args.resume_path:
        config['resume_path'] = args.resume_path
    if args.pretraining_path:
        config['pretraining_path'] = args.pretraining_path
    if args.exp_name:
        config['exp_name'] = args.exp_name
    path = os.path.join(config["working_dir"], f"{config['exp_name']}_{config['datetime']}")
    config['working_path'] = path
    print(f"Local Rank: {os.environ.get('LOCAL_RANK')}, working_path: {config['working_path']}")
    #################################################################
    if os.environ.get("LOCAL_RANK", 0) == 0:
        print(
            "\n" + "\n".join(list(map(lambda x: f"{x[0]:20}: {x[1]}", config.items())))
        )
    os.makedirs("./tmpdir", exist_ok=True)
    model_points_name = config['model_points'] if 'model_points' in config else 'minkunet'
    model_points_name = model_points_name.lower()
    dm = DownstreamDataModule(config)
    model = make_model(config, config["pretraining_path"])
    if config["num_gpus"] > 1:
        if model_points_name == 'minkunet':
            model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
        elif model_points_name == 'voxelnet':
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        elif model_points_name == 'cylinder3d':
            model = None
        elif model_points_name == 'cylinder3d_separate':
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #########################
    #######   module  #######
    if model_points_name == 'minkunet':
        module = LightningDownstream(model, config)
    elif model_points_name == 'voxelnet':
        module = LightningDownstreamSpconv(model, config)
    elif model_points_name == 'cylinder3d':
        module = LightningDownstreamSpconv(model, config)
    elif model_points_name == 'cylinder3d_separate':
        module = LightningDownstreamCylinder3D(model, config)
    else:
        raise Exception("Unknown model name")

    ## 历史遗留问题。之后，要把此处删除
    if model_points_name == 'cylinder3d':
        if config['pretraining_path'] is not None and os.path.exists(config['pretraining_path']):
            module.load_pretraining_file(config['pretraining_path'])

    tb_logger, csv_logger = TensorBoardLogger(save_dir=path), CSVLogger(save_dir=path)
    print(f"Fine-tuning the pre-trained 3d backbone {model_points_name} on panoptic segmentation task")
    if pl_version == 1:
        trainer = pl.Trainer(
            # common
            default_root_dir=path,
            max_epochs=config["num_epochs"],
            num_sanity_val_steps=0,
            check_val_every_n_epoch=1,
            logger=[tb_logger, csv_logger],
            gpus=config["num_gpus"],
            accelerator="ddp",
            checkpoint_callback=True,
            plugins=DDPPlugin(find_unused_parameters=False),
            resume_from_checkpoint=config["resume_path"],
        )
        print("Starting the training")
        print(f"path:{os.path.abspath(path)}")
        trainer.fit(module, dm)
        print(f"Training finished. Working dir is {os.path.abspath(path)}")
    elif pl_version == 2:
        from lightning.pytorch.strategies import DDPStrategy
        ddp = DDPStrategy(find_unused_parameters=False)  # 从nccl -> gloo
        val_interval = config['val_interval'] if 'val_interval' in config else 1
        trainer = pl.Trainer(
            # common
            default_root_dir=path,
            enable_checkpointing=True,
            max_epochs=config["num_epochs"],
            num_sanity_val_steps=0,
            check_val_every_n_epoch=val_interval,
            logger=[tb_logger, csv_logger],
            # v2.0
            devices=config["num_gpus"],
            accelerator="gpu",
            strategy=ddp,
        )
        print("Starting the training")
        print(f"path:{os.path.abspath(path)}")
        trainer.fit(module, dm, ckpt_path=config["resume_path"])
        print(f"Training finished. Working dir is {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
