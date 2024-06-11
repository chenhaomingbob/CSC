import argparse
import gc
import os
import torch

torch.set_float32_matmul_precision('high')
import MinkowskiEngine as ME

########### lightning v2.x #################
import pytorch_lightning

if pytorch_lightning.__version__.startswith("1"):
    pl_version = 1
    min_version = pytorch_lightning.__version__.split('.')[1]
    import pytorch_lightning as pl
    from pytorch_lightning.plugins import DDPPlugin
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
elif pytorch_lightning.__version__.startswith("2"):
    pl_version = 2
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from downstream.semantic_segmentation.lightning_datamodule import DownstreamDataModule
from downstream.semantic_segmentation.lightning_trainer import LightningDownstream
from downstream.semantic_segmentation.lightning_trainer_spconv import LightningDownstreamSpconv
from downstream.semantic_segmentation.model_builder import make_model
from utils.read_config import generate_config
import torch.nn as nn


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
        "--exp_name", type=str, default=None, help="provide a path to pre-trained weights"
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
    #################################################################
    if os.environ.get("LOCAL_RANK", 0) == 0:
        print(
            "\n" + "\n".join(list(map(lambda x: f"{x[0]:20}: {x[1]}", config.items())))
        )

    model_points_name = config['model_points'] if 'model_points' in config else 'minkunet'

    if model_points_name == 'minkunet':
        model_type = 'mink'
    elif model_points_name == 'voxelnet':
        model_type = 'spconv'
    else:
        raise Exception(f"Unkonw 3d model: {model_points_name}")

    dm = DownstreamDataModule(config)
    model = make_model(config, config["pretraining_path"])
    if config["num_gpus"] > 1:
        if model_type == 'mink':
            model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
        elif model_type == 'spconv':
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if model_type == 'mink':
        module = LightningDownstream(model, config)
    elif model_type == 'spconv':
        module = LightningDownstreamSpconv(model, config)

    tb_logger, csv_logger = TensorBoardLogger(save_dir=path), CSVLogger(save_dir=path)
    print(f"Fine-tuning the pre-trained 3d backbone {model_points_name} on semantic segmentation task")
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
            # num_sanity_val_steps=0,
            resume_from_checkpoint=config["resume_path"],
        )
        print("Starting the training")
        print(f"path:{os.path.abspath(path)}")
        trainer.fit(module, dm)
        print(f"Training finished. Working dir is {os.path.abspath(path)}")
    elif pl_version == 2:
        trainer = pl.Trainer(
            # common
            default_root_dir=path,
            enable_checkpointing=True,
            max_epochs=config["num_epochs"],
            num_sanity_val_steps=0,
            check_val_every_n_epoch=1,  #### TOOD
            logger=[tb_logger, csv_logger],
            # v2.0
            devices=config["num_gpus"],
            accelerator="gpu",
            # v1.0
            # gpus=config["num_gpus"],
            # accelerator="ddp",
            # checkpoint_callback=True,
            # plugins=DDPPlugin(find_unused_parameters=False),
            # num_sanity_val_steps=0,
            # resume_from_checkpoint=config["resume_path"],
            # check_val_every_n_epoch=100,
        )
        print("Starting the training")
        print(f"path:{os.path.abspath(path)}")
        trainer.fit(module, dm, ckpt_path=config["resume_path"])
        print(f"Training finished. Working dir is {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
