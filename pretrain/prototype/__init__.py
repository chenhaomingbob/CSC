from .lightning_datamodule_cluster import PretrainDataModule as PretrainDataModuleCluster
############################################### MinkUnet ###############################################
from .lightning_trainer_cluster_prototype_v1_1 import \
    LightningPretrain as LightningPretrain_cluster_v1_1
############################################### voxelnet ###############################################
from .lightning_trainer_cluster_prototype_v1_1_spconv import \
    LightningPretrainSpconv as LightningPretrainSpconv_cluster_v1_1
############################################### cylinder3d ###############################################
from .lightning_trainer_cluster_prototype_v1_1_cylinder3d import \
    LightningPretrainCylinder3D as LightningPretrainCylinder3D_cluster_v1_1

###########################################################################################################
prototype_dict = {
    ######## MinkUnet ########
    "cluster_prototype_v1_1": (LightningPretrain_cluster_v1_1, PretrainDataModuleCluster),
    ######## VoxelNet ########
    "cluster_prototype_v1_1_spconv": (LightningPretrainSpconv_cluster_v1_1, PretrainDataModuleCluster),
    ######## Cylinder3D ########
    "cluster_prototype_v1_1_cylinder3d": (LightningPretrainCylinder3D_cluster_v1_1, PretrainDataModuleCluster),

}
