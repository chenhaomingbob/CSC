import torch
from model import (
    MinkUNet,
    VoxelNet
)


def load_state_with_same_shape(model, weights):
    """
    Load common weights in two similar models
    (for instance between a pretraining and a downstream training)
    """
    model_state = model.state_dict()
    if list(weights.keys())[0].startswith("model."):
        weights = {k.partition("model.")[2]: weights[k] for k in weights.keys()}

    if list(weights.keys())[0].startswith("model_points."):
        weights = {k.partition("model_points.")[2]: weights[k] for k in weights.keys()}

    if list(weights.keys())[0].startswith("module."):
        print("Loading multigpu weights with module. prefix...")
        weights = {k.partition("module.")[2]: weights[k] for k in weights.keys()}

    if list(weights.keys())[0].startswith("encoder."):
        print("Loading multigpu weights with encoder. prefix...")
        weights = {k.partition("encoder.")[2]: weights[k] for k in weights.keys()}

    if list(weights.keys())[0].startswith('model_q.'):
        print("Loading model weights with model_q. prefix..")
        new_weights = {}
        for k in weights.keys():
            if k.startswith('model_q'):
                new_weights[k.partition("model_q.")[2]] = weights[k]
        weights = new_weights
        # weights = {k.partition("encoder.")[2]: weights[k] for k in weights.keys()}

    filtered_weights = {
        k: v
        for k, v in weights.items()
        if (k in model_state and v.size() == model_state[k].size())
    }
    removed_weights = {
        k: v
        for k, v in weights.items()
        if not (k in model_state and v.size() == model_state[k].size())
    }
    print("Loading weights:" + ", ".join(filtered_weights.keys()))
    print("")
    print("Not loading weights:" + ", ".join(removed_weights.keys()))
    return filtered_weights


def make_model(config, load_path=None):
    """
    Build the points model according to what is in the config
    """
    assert not config[
        "normalize_features"
    ], "You shouldn't normalize features for the downstream task"

    if '3d_model' in config:
        if config['3d_model'] == 'MinkUnet18A':
            from model.mink_unet import MinkUNet18A
            model = MinkUNet18A(in_channels=3, out_channels=768)
            # model = MinkUNet18A(in_channels=3, out_channels=config["model_n_out"])
            if load_path:
                checkpoint = torch.load(load_path, map_location="cpu")
                weights = checkpoint['state_dict']
                new_weights = {}
                for k in weights.keys():
                    if k.startswith('net3d'):
                        new_key = k.partition("net3d.")[2]
                        # if not new_key.startswith("final"):
                        #     new_weights[new_key] = weights[k]
                        new_weights[new_key] = weights[k]
                weights = new_weights
                model.load_state_dict(weights, strict=True)
    else:
        # 默认使用 minkunet
        model_points_name = 'minkunet'
        if 'model_points' in config:
            model_points_name = config['model_points']
        ## 如果是minkunet
        if model_points_name == 'minkunet':
            model = MinkUNet(1, config["model_n_out"], config)
            if load_path:
                print("Training with pretrained model")
                checkpoint = torch.load(load_path, map_location="cpu")
                if "config" in checkpoint:
                    for cfg in ("voxel_size", "cylindrical_coordinates"):
                        assert checkpoint["config"][cfg] == config[cfg], (
                            f"{cfg} is not consistant. "
                            f"Checkpoint: {checkpoint['config'][cfg]}, "
                            f"Config: {config[cfg]}."
                        )
                if set(checkpoint.keys()) == set(["epoch", "model", "optimizer", "train_criterion"]):
                    print("Pre-trained weights are coming from DepthContrast.")
                    pretraining_epochs = checkpoint["epoch"]
                    print(f"==> Number of pre-training epochs {pretraining_epochs}")
                    checkpoint = checkpoint["model"]
                    if list(checkpoint.keys())[0].startswith("module."):
                        print("Loading multigpu weights with module. prefix...")
                        checkpoint = {k.partition("module.")[2]: checkpoint[k] for k in checkpoint.keys()}
                    voxel_net_suffix = "trunk.2."
                    checkpoint = {
                        key.partition(voxel_net_suffix)[2]: checkpoint[key]
                        for key in checkpoint.keys() if key.startswith(voxel_net_suffix)
                    }
                    print(f"==> Number of loaded weight blobs {len(checkpoint)}")
                    checkpoint = {"model_points": checkpoint}
                key = "model_points" if "model_points" in checkpoint else "state_dict"
                filtered_weights = load_state_with_same_shape(model, checkpoint[key])
                model_dict = model.state_dict()
                model_dict.update(filtered_weights)
                model.load_state_dict(model_dict)
        ## 如果是voxelnet
        elif model_points_name == 'voxelnet':
            model = VoxelNet(4, config['model_n_out'], config)
            if load_path:
                print("Training with pretrained model")
                checkpoint = torch.load(load_path, map_location="cpu")
                # 检查预训练和微调时voxel_size是否一致
                if "config" in checkpoint:
                    for cfg in ("voxel_size", "cylindrical_coordinates"):
                        assert checkpoint["config"][cfg] == config[cfg], (
                            f"{cfg} is not consistant. "
                            f"Checkpoint: {checkpoint['config'][cfg]}, "
                            f"Config: {config[cfg]}."
                        )
                if set(checkpoint.keys()) == set(["epoch", "model", "optimizer", "train_criterion"]):
                    print("Pre-trained weights are coming from DepthContrast.")
                    pretraining_epochs = checkpoint["epoch"]
                    print(f"==> Number of pre-training epochs {pretraining_epochs}")
                    checkpoint = checkpoint["model"]
                    if list(checkpoint.keys())[0].startswith("module."):
                        print("Loading multigpu weights with module. prefix...")
                        checkpoint = {k.partition("module.")[2]: checkpoint[k] for k in checkpoint.keys()}
                    voxel_net_suffix = "trunk.2."
                    checkpoint = {
                        key.partition(voxel_net_suffix)[2]: checkpoint[key]
                        for key in checkpoint.keys() if key.startswith(voxel_net_suffix)
                    }
                    print(f"==> Number of loaded weight blobs {len(checkpoint)}")
                    checkpoint = {"model_points": checkpoint}
                key = "model_points" if "model_points" in checkpoint else "state_dict"
                filtered_weights = load_state_with_same_shape(model, checkpoint[key])
                model_dict = model.state_dict()
                model_dict.update(filtered_weights)
                model.load_state_dict(model_dict)
    if config["freeze_layers"]:
        # linear probing
        for param in list(model.parameters())[:-2]:
            param.requires_grad = False
    return model
