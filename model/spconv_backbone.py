from functools import partial
import numpy as np

import spconv.pytorch as spconv
import torch.nn as nn


def post_act_block(
        in_channels,
        out_channels,
        kernel_size,
        indice_key=None,
        stride=1,
        padding=0,
        conv_type="subm",
        norm_fn=None,
):
    if conv_type == "subm":
        conv = spconv.SubMConv3d(
            in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key
        )
    elif conv_type == "spconv":
        conv = spconv.SparseConv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            indice_key=indice_key,
        )
    elif conv_type == "inverseconv":
        conv = spconv.SparseInverseConv3d(
            in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False
        )
    elif conv_type == "transposeconv":
        conv = spconv.SparseConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False, indice_key=indice_key
        )
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
            self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None
    ):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity.features
        out.features = self.relu(out.features)

        return out


class VoxelBackBone8x(nn.Module):
    def __init__(self, input_channels, grid_size, **kwargs):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_channels, 16, 3, padding=1, bias=False, indice_key="subm1"
            ),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key="subm1"),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(
                16,
                32,
                3,
                norm_fn=norm_fn,
                stride=2,
                padding=1,
                indice_key="spconv2",
                conv_type="spconv",
            ),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key="subm2"),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key="subm2"),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(
                32,
                64,
                3,
                norm_fn=norm_fn,
                stride=2,
                padding=1,
                indice_key="spconv3",
                conv_type="spconv",
            ),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key="subm3"),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key="subm3"),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(
                64,
                64,
                3,
                norm_fn=norm_fn,
                stride=2,
                padding=(0, 1, 1),
                indice_key="spconv4",
                conv_type="spconv",
            ),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key="subm4"),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key="subm4"),
        )

        last_pad = 0
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(
                64,
                128,
                (3, 1, 1),
                stride=(2, 1, 1),
                padding=last_pad,
                bias=False,
                indice_key="spconv_down2",
            ),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            "x_conv1": 16,
            "x_conv2": 32,
            "x_conv3": 64,
            "x_conv4": 64,
        }

    def forward(self, input_sp_tensor):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        x = self.conv_input(input_sp_tensor)  # (40,1024,1024)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        out = self.conv_out(x_conv4)
        return out


class VoxelResBackBone8x(nn.Module):
    def __init__(self, input_channels, grid_size, **kwargs):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_channels, 16, 3, padding=1, bias=False, indice_key="subm1"
            ),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key="res1"),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key="res1"),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(
                16,
                32,
                3,
                norm_fn=norm_fn,
                stride=2,
                padding=1,
                indice_key="spconv2",
                conv_type="spconv",
            ),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key="res2"),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key="res2"),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(
                32,
                64,
                3,
                norm_fn=norm_fn,
                stride=2,
                padding=1,
                indice_key="spconv3",
                conv_type="spconv",
            ),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key="res3"),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key="res3"),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(
                64,
                128,
                3,
                norm_fn=norm_fn,
                stride=2,
                padding=(0, 1, 1),
                indice_key="spconv4",
                conv_type="spconv",
            ),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key="res4"),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key="res4"),
        )

        last_pad = 0
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(
                128,
                128,
                (3, 1, 1),
                stride=(2, 1, 1),
                padding=last_pad,
                bias=False,
                indice_key="spconv_down2",
            ),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            "x_conv1": 16,
            "x_conv2": 32,
            "x_conv3": 64,
            "x_conv4": 128,
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = (
            batch_dict["voxel_features"],
            batch_dict["voxel_coords"],
        )
        batch_size = batch_dict["batch_size"]
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size,
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update(
            {"encoded_spconv_tensor": out, "encoded_spconv_tensor_stride": 8}
        )
        batch_dict.update(
            {
                "multi_scale_3d_features": {
                    "x_conv1": x_conv1,
                    "x_conv2": x_conv2,
                    "x_conv3": x_conv3,
                    "x_conv4": x_conv4,
                }
            }
        )

        return batch_dict


class HeightCompression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, encoded_spconv_tensor):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        # encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        return spatial_features


class VoxelNet(VoxelBackBone8x):
    def __init__(self, in_channels, out_channels, config, D=3):
        self.bev_stride = 8
        voxel_size = [0.1, 0.1, 0.2]  # nuScenes
        point_cloud_range = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], dtype=np.float32)  # nuScenes
        self.grid_size = ((point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size).astype(int)[::-1]  # 这边转置了
        #
        self.bach_size = config["batch_size"]
        super().__init__(in_channels, self.grid_size)

        self.without_compression = config['without_compression'] if 'without_compression' in config else False
        self.height_compression = HeightCompression()
        ####
        self.upsample = config['upsample'] if 'upsample' in config else False

        if self.upsample:
            self.decoder = self._make_decoder(config)

        self.final = spconv.SparseConv3d(
            64 if self.upsample else 128,
            out_channels // 1,
            1,
            stride=1,
            padding=0,
            bias=False,
            indice_key="final",
        )

    def _make_decoder(self, config):
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if self.upsample == 2:
            return spconv.SparseSequential(
                spconv.SparseInverseConv3d(128, 64, (3, 1, 1), bias=True,
                                           indice_key="spconv_down2"),
                norm_fn(64),
                nn.ReLU(),
            )
        elif self.upsample == 4:
            return spconv.SparseSequential(
                spconv.SparseInverseConv3d(128, 64, (3, 1, 1), bias=True, indice_key="spconv_down2"),
                norm_fn(64),
                nn.ReLU(),
                spconv.SparseInverseConv3d(64, 64, 3, bias=True, indice_key="spconv4"),
                norm_fn(64),
                nn.ReLU(),
            )
        elif self.upsample == 8:
            return spconv.SparseSequential(
                spconv.SparseInverseConv3d(128, 64, (3, 1, 1), bias=True, indice_key="spconv_down2"),
                norm_fn(64),
                nn.ReLU(),
                spconv.SparseInverseConv3d(64, 64, 3, bias=True, indice_key="spconv4"),
                norm_fn(64),
                nn.ReLU(),
                spconv.SparseInverseConv3d(64, 64, 3, bias=True, indice_key="spconv3"),
                norm_fn(64),
                nn.ReLU(),
            )
        elif self.upsample == 16:
            return spconv.SparseSequential(
                spconv.SparseInverseConv3d(128, 128, (3, 1, 1), bias=True, indice_key="spconv_down2"),
                norm_fn(128),
                nn.ReLU(),
                spconv.SparseInverseConv3d(128, 64, 3, bias=True, indice_key="spconv4"),
                norm_fn(64),
                nn.ReLU(),
                spconv.SparseInverseConv3d(64, 64, 3, bias=True, indice_key="spconv3"),
                norm_fn(64),
                nn.ReLU(),
                spconv.SparseInverseConv3d(64, 64, 3, bias=True, indice_key="spconv2"),
                norm_fn(64),
                nn.ReLU(),
            )

    def forward(self, voxels, coordinates):
        sp_tensor = spconv.SparseConvTensor(
            features=voxels,  # (117148,4)
            indices=coordinates,
            spatial_shape=self.grid_size,  # (40,1024,1024)
            batch_size=self.bach_size
        )
        sp_tensor = super(VoxelNet, self).forward(sp_tensor)  # (20755,128)

        if hasattr(self, 'decoder'):
            sp_tensor = self.decoder(sp_tensor)

        sp_tensor = self.final(sp_tensor)
        sp_tensor = sp_tensor.replace_feature(nn.functional.normalize(sp_tensor.features, dim=1))

        if not self.without_compression:
            sp_tensor = self.height_compression(sp_tensor)
        return sp_tensor