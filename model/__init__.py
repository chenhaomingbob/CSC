from model.image_model import *
# from model.maskclip_model import maskClipFeatureExtractor
from model.cylinder3d.cylinder3D import Cylinder3D

try:
    from model.res16unet import Res16UNet34C as MinkUNet
except ImportError:
    MinkUNet = None
try:
    from model.spconv_backbone import VoxelNet
except (ImportError, AttributeError):
    VoxelNet = None
