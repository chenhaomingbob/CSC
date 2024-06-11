from model import (
    MinkUNet,
    VoxelNet,
    DilationFeatureExtractor,
    PPKTFeatureExtractor,
    Preprocessing,
    DinoVitFeatureExtractor,
    # maskClipFeatureExtractor,
    Cylinder3D,
)
from loguru import logger


def forgiving_state_restore(net, loaded_dict):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """
    loaded_dict = {
        k.replace("module.", ""): v for k, v in loaded_dict.items()
    }
    net_state_dict = net.state_dict()
    new_loaded_dict = {}
    for k in net_state_dict:
        new_k = k
        if (
                new_k in loaded_dict and net_state_dict[k].size() == loaded_dict[new_k].size()
        ):
            new_loaded_dict[k] = loaded_dict[new_k]
        else:
            print("Skipped loading parameter {}".format(k))
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)
    return net


def make_model(config):
    """
    Build points and image models according to what is in the config
    """
    ###########################
    ## 3D point cloud model  ##
    ###########################
    model_points_name = config["model_points"]
    assert model_points_name in ["voxelnet", "minkunet", "cylinder3d", "cylinder3d_separate"]
    if model_points_name == "voxelnet":
        model_points = VoxelNet(4, config["model_n_out"], config)
    elif model_points_name == 'minkunet':
        model_points = MinkUNet(1, config["model_n_out"], config)
    elif model_points_name == 'cylinder3d' or model_points_name == 'cylinder3d_separate':
        model_points = Cylinder3D(input_dim=4, output_dim=config["model_n_out"], cfg=config)
    logger.info(f"finish build model points: {model_points_name} ")
    ###########################
    ##     2D image model    ##
    ###########################
    if config["images_encoder"].find("vit_") != -1:
        model_images = DinoVitFeatureExtractor(config, preprocessing=Preprocessing())
    elif config["decoder"] == "dilation":
        model_images = DilationFeatureExtractor(config, preprocessing=Preprocessing())
    elif config["decoder"] == "ppkt":
        model_images = PPKTFeatureExtractor(config, preprocessing=Preprocessing())
    else:
        raise Exception(f"Model not found: {config['decoder']}")
    logger.info(f"finish build model image: {config['decoder']} ")
    ########################
    if 'pretraining_path' in config:
        import torch
        load_path = config['pretraining_path']
        checkpoint = torch.load(load_path, map_location="cpu")
        model_points.load_state_dict(checkpoint["model_points"], strict=True)
        model_images.load_state_dict(checkpoint["model_images"], strict=True)
    ########################
    return model_points, model_images
