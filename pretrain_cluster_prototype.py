import argparse
import os
import shutil
import time

import MinkowskiEngine as ME
import torch
import torch.nn as nn

import pytorch_lightning

pl_version = 1
if pytorch_lightning.__version__.startswith("1"):
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
    from pytorch_lightning.plugins import DDPPlugin
elif pytorch_lightning.__version__.startswith("2"):
    pl_version = 2
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
    from lightning.pytorch.strategies import DDPStrategy
    # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html#earlystopping
from pretrain import __all__ as pretrain_funs
from pretrain.model_builder import make_model
from utils.read_config import generate_config
from loguru import logger
print(f"pytorch_lightning version is: {pl_version}")


def main():
    """
    Code for launching the pretraining
    """
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="config/slidr_minkunet.yaml",
        help="specify the config for training",
        required=True
    )
    parser.add_argument(
        "--resume_path", type=str, default=None, help="provide a path to resume an incomplete training"
    )
    parser.add_argument(
        "--pretraining_path", type=str, default=None, help="provide a path to pre-trained weights"
    )

    args = parser.parse_args()
    config = generate_config(args.cfg_file)
    if args.resume_path:
        config['resume_path'] = args.resume_path
    if args.pretraining_path:
        config['pretraining_path'] = args.pretraining_path
    root_dir = os.path.join(config["working_dir"], config["datetime"])
    print(root_dir)
    os.makedirs(root_dir, exist_ok=True)
    config['working_dir'] = root_dir

    if os.environ.get("LOCAL_RANK", 0) == 0:
        print("\n" + "\n".join(list(map(lambda x: f"{x[0]:20}: {x[1]}", config.items()))))
    # ========================================================================================
    model_points_name = config["model_points"]
    if model_points_name == "minkunet":
        LightningPretrain, PretrainDataModule = pretrain_funs[config['exp_name']]
    elif model_points_name == "voxelnet":
        LightningPretrainSpconv, PretrainDataModule = pretrain_funs[config['exp_name']]
    elif model_points_name == 'cylinder3d':
        LightningPretrainCylinder3D, PretrainDataModule = pretrain_funs[config['exp_name']]
    elif model_points_name == 'cylinder3d_separate':
        LightningPretrainCylinder3D, PretrainDataModule = pretrain_funs[config['exp_name']]
    # ========================================================================================
    logger.info("Making model...")
    data_loader = PretrainDataModule(config)  # dataloader
    model_points, model_images = make_model(config)
    if config["num_gpus"] > 1:
        if model_points_name == "minkunet":
            model_points = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model_points)
        elif model_points_name == "voxelnet":
            model_points = nn.SyncBatchNorm.convert_sync_batchnorm(model_points)
        elif model_points_name == 'cylinder3d':
            model_points = None
        elif model_points_name == 'cylinder3d_separate':
            model_points = nn.SyncBatchNorm.convert_sync_batchnorm(model_points)
        # if model_points
        model_images = nn.SyncBatchNorm.convert_sync_batchnorm(model_images)
    # ========================================================================================
    ###### 选择模型
    logger.info("Making module...")
    if model_points_name == "minkunet":
        module = LightningPretrain(model_points, model_images, config)
    elif model_points_name == "voxelnet":
        module = LightningPretrainSpconv(model_points, model_images, config)
    elif model_points_name == 'cylinder3d' or model_points_name == 'cylinder3d_separate':
        module = LightningPretrainCylinder3D(model_points, model_images, config)

    ############
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=os.path.join(root_dir, 'checkpoints'))
    #################################
    file_dir = os.path.join(root_dir, 'pretrain_files')
    class_path = str(type(module)).split("'")[1].rsplit(".", maxsplit=1)[0].replace(".", "/") + ".py"
    class_name = class_path.split("/")[-1]
    os.makedirs(file_dir, exist_ok=True)
    # 复制 LightningModule 文件
    shutil.copy2(
        src=os.path.join(os.path.join(os.path.abspath('.'), class_path)),
        dst=os.path.join(os.path.abspath('.', ), os.path.join(file_dir, class_name))
    )
    # 复制 配置文件
    shutil.copy2(
        src=os.path.join(os.path.join(os.path.abspath('.'), args.cfg_file)),
        dst=os.path.join(os.path.abspath('.', ), os.path.join(file_dir, args.cfg_file.split("/")[-1]))
    )
    ################################
    tb_logger = TensorBoardLogger(save_dir=os.path.join(root_dir, "tb_logs"))
    csv_logger = CSVLogger(save_dir=os.path.join(root_dir, "csv_logs"))

    print("##################################################")
    print(f"Pre-training the 3d backbone {model_points_name}. \n"
          f"Pytorch Version:{torch.__version__}; Lightning Version:{pl.__version__}")
    print("##################################################")
    start_time = time.time()
    if pl_version == 1:
        trainer = pl.Trainer(
            # common
            default_root_dir=root_dir,
            max_epochs=config["num_epochs"],
            num_sanity_val_steps=0,
            check_val_every_n_epoch=config['cluster_interval'],
            log_every_n_steps=10,
            callbacks=[checkpoint_callback],
            gpus=config["num_gpus"],
            accelerator="ddp",
            checkpoint_callback=True,
            resume_from_checkpoint=config["resume_path"],
            plugins=DDPPlugin(find_unused_parameters=True))
        trainer.fit(module, data_loader)
        # print(f"Training finished. Working dir is {os.path.abspath(root_dir)}")
    elif pl_version == 2:
        from datetime import datetime, timedelta
        # ddp = DDPStrategy(find_unused_parameters=True)  # 从nccl -> gloo
        # ddp = DDPStrategy(process_group_backend='nccl', find_unused_parameters=True)  # 从nccl -> gloo
        ddp = DDPStrategy(process_group_backend='gloo', find_unused_parameters=True)  # 从nccl -> gloo
        trainer = pl.Trainer(
            # common
            default_root_dir=root_dir,
            enable_checkpointing=True,
            max_epochs=config["num_epochs"],
            num_sanity_val_steps=0,  # 在训练之前，运行batch n个验证步骤.
            check_val_every_n_epoch=config['cluster_interval'],
            # v2.0,
            devices=config["num_gpus"],
            accelerator="gpu",
            callbacks=[checkpoint_callback],
            logger=[tb_logger, csv_logger],
            log_every_n_steps=10,
            strategy=ddp
        )
        trainer.fit(module, data_loader, ckpt_path=config["resume_path"])
    run_time = time.time() - start_time
    hours, mins = int(run_time // 3600), int((run_time % 3600) // 60)
    print(f"Training finished. "
          f"Runtime Duration is {str(hours)}H{str(mins)}M. "
          f"Working dir is {os.path.abspath(root_dir)}")


if __name__ == "__main__":
    main()
