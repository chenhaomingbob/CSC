import torch
from model import MinkUNet
from model.panoptic_models.panoptic_polarnet import Panoptic_PolarNet


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

    panoptic_model = config['panoptic_model']
    model_points_name = config['model_points'] if 'model_points' in config else 'minkunet'
    model_points_name = model_points_name.lower()
    if panoptic_model == 'panoptic_polarnet':
        model = Panoptic_PolarNet(config)
        if model_points_name == 'cylinder3d_separate' and load_path:
            if load_path:
                checkpoint = torch.load(load_path, map_location="cpu")
                model.backbone_3d.load_state_dict(checkpoint['cylinder_3d_generator'], strict=True)
                weights = checkpoint['cylinder_3d_spconv_seg']
                del weights['logits.weight']
                del weights['logits.bias']
                model.voxel_sem_head.load_state_dict(weights, strict=False)
                print("Not loading weights: logits.weight, logits.bias")
    if config["freeze_layers"]:
        # linear probing 仅训练最后的fc layer
        for param in list(model.parameters())[:-2]:
            param.requires_grad = False
    return model
