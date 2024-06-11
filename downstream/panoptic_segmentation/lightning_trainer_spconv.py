import os
import yaml
import json
import pytorch_lightning
import numpy as np

if pytorch_lightning.__version__.startswith("1"):
    import pytorch_lightning as pl
    from pytorch_lightning.utilities import rank_zero_only
elif pytorch_lightning.__version__.startswith("2"):
    import lightning.pytorch as pl
    from lightning.pytorch.utilities import rank_zero_only
import torch
import torch.optim as optim
import pickle
from model.panoptic_models.sub_layers import cylinder_fea, Asymm_3d_spconv, BEV_Unet

from downstream.panoptic_segmentation.criterion import DownstreamLoss
from downstream.panoptic_segmentation.post_processing import get_panoptic_segmentation
from downstream.panoptic_segmentation.lovasz_losses import lovasz_softmax
from downstream.panoptic_segmentation.eval_pq import PanopticEval
from downstream.panoptic_segmentation.dataset_utils import collate_dataset_info


class LightningDownstreamSpconv(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.save_hyperparameters()
        # self.model = model
        self.best_PQ = 0.0
        # self.metrics = {"val mIoU": [], "val_loss": [], "train_loss": []}
        self._config = config
        self.train_losses = []

        self.val_losses = []
        self.ignore_index = config["ignore_index"]
        self.n_classes = config["model_n_out"]
        self.epoch = 0
        if config["loss"].lower() == "lovasz":
            self.criterion = DownstreamLoss(
                ignore_index=config["ignore_index"],
                device=self.device,
            )  # a lovasz loss and a crossentropy
        else:
            self.criterion = torch.nn.CrossEntropyLoss(
                ignore_index=config["ignore_index"],
            )
        ##########################
        self.working_dir = self.hparams['config']['working_path']
        self.temp_dir = os.path.join(self.working_dir, "tmpdir")
        self.make_dir()
        ###
        ##
        self.save_hyperparameters('config')
        self.validation_step_outputs = []

        self.model_point_name = config['model_points'].lower()
        ######
        self.grid_size = [480, 360, 32]  #
        self.output_dim_3d_backbone = config['output_dim_model_points']  # 20 for nuScenes
        self.model_point_name = config['model_points'].lower()  #
        self.num_classes = config['model_n_out']  # 17 if dataset == 'nuscenes'

        if self.model_point_name == 'minkunet':
            raise Exception("Not implementation")
        elif self.model_point_name == 'voxelnet':
            raise Exception("Not implementation")
        elif self.model_point_name == 'cylinder3d':
            model_cfg = config

            self.pure_version = model_cfg['pure'] if 'pure' in model_cfg else False
            ##############
            self.cylinder_3d_generator = cylinder_fea(
                model_cfg,
                grid_size=self.grid_size,
                fea_dim=4,
                out_pt_fea_dim=128,  # 256->128
                fea_compre=16
            )

            self.cylinder_3d_spconv_seg = Asymm_3d_spconv(
                model_cfg,
                output_shape=self.grid_size,
                use_norm=True,
                num_input_features=16,
                init_size=32,
                nclasses=self.num_classes
            )
            ##########
            self.UNet = BEV_Unet(input_dim=128, output_dim=1, pure_version=self.pure_version)
        else:
            raise Exception("Unknown 3d backbone")
        dataset_name = 'nuscenes'
        self.unique_label, self.unique_label_str, self.thing_list = collate_dataset_info(dataset_name)
        self.denoise_thing_list = [i - 1 for i in self.thing_list]
        self.ignore_label = len(self.unique_label)  # nuScenes: 16
        # nuscenes
        num_classes = 17  # 17 , 算上noise类, 共17个
        self.panoptic_evaluator = PanopticEval(
            num_classes,
            None,
            [self.ignore_label],  #
            min_points=self._config['panoptic_eval_min_points'],
        )

        self.ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_label)
        self.center_loss_fn = torch.nn.MSELoss()
        self.offset_loss_fn = torch.nn.L1Loss()
        self.center_loss_weight = 100
        self.offset_loss_weight = 10
        #####
        self.point_sem_pred_list, self.point_inst_pred_list, self.point_sem_gt_list, self.point_inst_gt_list = [], [], [], []

    @rank_zero_only
    def make_dir(self):
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        config_save_path = os.path.join(self.working_dir, 'config_file.yaml')
        with open(config_save_path, 'w') as f:
            yaml.dump(self._config, f)

        print(f"working_dir {os.path.abspath(self.working_dir)}")
        print(f"temp_dir {os.path.abspath(self.temp_dir)}")

    def load_pretraining_file(self, pretraining_path):
        checkpoint = torch.load(pretraining_path, map_location='cpu')
        self.cylinder_3d_generator.load_state_dict(checkpoint['cylinder_3d_generator'], strict=True)

        weights = checkpoint['cylinder_3d_spconv_seg']
        del weights['logits.weight']
        del weights['logits.bias']
        self.cylinder_3d_spconv_seg.load_state_dict(weights, strict=False)
        print("Not loading weights: logits.weight, logits.bias")

    def configure_optimizers(self):
        # Optimizer
        use_different_lr = True if 'pretraining_path' in self._config and 'lr_head' in self._config else False
        optimizer_name = self._config.get("optimizer").lower() if 'optimizer' in self._config else 'SGD'
        if use_different_lr:
            # if self._config.get("lr_head", None) is not None:
            print("Use different learning rates between the head and trunk.")

            def is_cylinder_logits(key):
                return key.find('logits.') == 0

            all_parameters = list(self.cylinder_3d_generator.parameters()) + list(
                self.cylinder_3d_spconv_seg.parameters()) + list(self.UNet.parameters())

            backbone_params = list(self.cylinder_3d_generator.parameters()) \
                              + [param for key, param in self.cylinder_3d_spconv_seg.named_parameters()
                                 if param.requires_grad and not is_cylinder_logits(key)]  #

            head_params = list(self.UNet.named_parameters()) \
                          + [(key, param) for key, param in self.cylinder_3d_spconv_seg.named_parameters()
                             if param.requires_grad and is_cylinder_logits(key)]

            assert len(all_parameters) == (len(head_params) + len(backbone_params))

            weight_decay = self._config["weight_decay"] if (
                    "weight_decay" in self._config) else None
            weight_decay_head = self._config["weight_decay_head"] if (
                    "weight_decay_head" in self._config) else weight_decay
            lr_head = self._config["lr_head"]
            lr_backbone = self._config['lr']

            parameters = [
                {
                    "params": iter(head_params),
                    "lr": lr_head,
                    "weight_decay": weight_decay_head
                },
                {
                    "params": iter(backbone_params)
                },
            ]
            print(
                f"==> Head:  #{len(head_params)} params with learning rate: {lr_head} and weight_decay: {weight_decay_head}")
            print(
                f"==> Trunk: #{len(backbone_params)} params with learning rate: {lr_backbone} and weight_decay: {weight_decay}")

            if optimizer_name == 'adam':
                print('Optimizer: Adam')
                optimizer = optim.Adam(
                    all_parameters,
                    lr=lr_backbone,
                    # weight_decay=self._config["weight_decay"],
                )
            elif optimizer_name == 'adamw':
                print('Optimizer: AdamW')
                optimizer = optim.AdamW(
                    all_parameters,
                    lr=lr_backbone,
                    weight_decay=weight_decay,
                )
            else:
                optimizer = optim.SGD(
                    parameters,
                    lr=lr_backbone,
                    momentum=self._config["sgd_momentum"],
                    dampening=self._config["sgd_dampening"],
                    weight_decay=weight_decay,
                )
        else:

            all_parameters = list(self.cylinder_3d_generator.parameters()) + list(
                self.cylinder_3d_spconv_seg.parameters()) + list(self.UNet.parameters())

            if optimizer_name == 'adam':
                print('Optimizer: Adam')
                optimizer = optim.Adam(
                    all_parameters,
                    lr=self._config["lr"],
                    # weight_decay=self._config["weight_decay"],
                )
            elif optimizer_name == 'adamw':
                print('Optimizer: AdamW')
                optimizer = optim.AdamW(
                    all_parameters,
                    lr=self._config["lr"],
                    weight_decay=self._config["weight_decay"],
                )
            else:
                print('Optimizer: SGD')
                optimizer = optim.SGD(
                    all_parameters,
                    lr=self._config["lr"],
                    momentum=self._config["sgd_momentum"],
                    dampening=self._config["sgd_dampening"],
                    weight_decay=self._config["weight_decay"],
                )
        scheduler_name = self._config.get("scheduler").lower() if 'scheduler' in self._config else 'cosine'
        if scheduler_name == 'multi_steplr':
            print('Scheduler: Multi-StepLR')
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[10, 20, 30], gamma=0.5
            )
        elif scheduler_name == 'steplr':
            print('Scheduler: StepLR')
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, int(.9 * self._config["num_epochs"]),
            )
        elif scheduler_name == 'cosine':
            print('Scheduler: Cosine')
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self._config["num_epochs"]
            )
        else:
            raise Exception("Unknow scheduler")

        return [optimizer], [scheduler]

    def forward_network(self, batch_dict):
        # pt_fea: [(122078,9)].  xy_ind_tensor: [(122078,3)]
        pt_fea, xy_ind_tensor = batch_dict["pc_features"], batch_dict['pc_grid_index']

        # 对每个网格的点进行池化
        coords, features_3d = self.cylinder_3d_generator(pt_fea, xy_ind_tensor, batch_dict)
        # 主干网络
        if self.pure_version:
            sem_pred, bev_features = self.cylinder_3d_spconv_seg(features_3d, coords, len(pt_fea))
            bev_features = bev_features.dense().squeeze(-1)  # [B,128,480,360]

            center_pred, offset_pred = self.UNet(bev_features)
        else:
            # 主干网络
            sem_pred, center_pred, offset_pred = self.cylinder_3d_spconv_seg(features_3d, coords, len(pt_fea))
            # 稠密heatmap网络
            center_pred = self.UNet(center_pred)

        output_dict = {
            'voxel_prediction': sem_pred,
            'pred_center': center_pred,
            'pred_offset': offset_pred
        }

        if self.training:
            total_loss, loss_dict = self.get_training_loss(output_dict, batch_dict)
            del batch_dict
            del output_dict
            return total_loss, loss_dict
        else:
            pred_dicts = self.post_processing(output_dict, batch_dict)
            del batch_dict
            del output_dict
            return pred_dicts

    def post_processing(self, output_dict, batch_dict):
        # GT data
        gt_points_grid_index = batch_dict['pc_grid_index']
        # gt_points_grid_index = [x.cpu().numpy() for x in batch_dict['pc_grid_index']]
        # gt_points_label: 0 denotes barrier and 16 denotes noise class. (做过noise变换，以及-1)
        gt_points_label = batch_dict['point_sem_label']
        gt_points_inst = batch_dict['point_inst_label']
        # gt_points_sem_label = batch_dict['evaluation_point_sem_label'] # 0 表示noise label and 16 denotes vegetation. (原始数据)
        # gt_points_inst_label = batch_dict['evaluation_point_inst_label']
        # prediction
        voxel_prediction = output_dict['voxel_prediction']  # (B,17,480,360,32)
        pred_center = output_dict['pred_center']
        pred_offset = output_dict['pred_offset']
        ###
        post_proc_threshold, post_proc_nms_kernel, post_proc_top_k, post_proc_use_polar = 0.1, 5, 100, True
        #
        voxel_panoptic_results, point_panoptic_results, point_gt_labels_list, point_gt_instances_list = [], [], [], []
        # sem_hist_list = []

        for cur_sample_index, cur_gt_points_grid_index in enumerate(gt_points_grid_index):
            ###### Panoptic Segmentation ######
            cur_gt_points_grid_index = cur_gt_points_grid_index.int()
            point_x_index = cur_gt_points_grid_index[:, 0]
            point_y_index = cur_gt_points_grid_index[:, 1]
            point_z_index = cur_gt_points_grid_index[:, 2]
            for_mask = torch.zeros(1, self.grid_size[0], self.grid_size[1], self.grid_size[2],
                                   dtype=torch.bool).cuda()
            for_mask[0, point_x_index, point_y_index, point_z_index] = True

            cur_voxel_prediction = voxel_prediction[cur_sample_index]  # 要求数值的范围在 [0~20]
            cur_center, cur_offset = pred_center[cur_sample_index], pred_offset[cur_sample_index]

            cur_voxel_panoptic, center_points = get_panoptic_segmentation(
                torch.unsqueeze(cur_voxel_prediction, 0),
                torch.unsqueeze(cur_center, 0),
                torch.unsqueeze(cur_offset, 0),
                self.denoise_thing_list,
                threshold=post_proc_threshold,
                nms_kernel=post_proc_nms_kernel,
                top_k=post_proc_top_k,
                polar=post_proc_use_polar,
                foreground_mask=for_mask
            )
            cur_voxel_panoptic = cur_voxel_panoptic.int()
            point_panoptic = cur_voxel_panoptic[0, point_x_index, point_y_index, point_z_index]
            ###########################
            ## panoptic segmentation ##
            ###########################
            voxel_panoptic_results.append(cur_voxel_panoptic.cpu().detach().numpy())
            point_panoptic_results.append(point_panoptic.cpu().detach().numpy())
            point_gt_labels_list.append(np.squeeze(gt_points_label[cur_sample_index]))  # point的 gt class id
            point_gt_instances_list.append(np.squeeze(gt_points_inst[cur_sample_index]))  # point的 gt instance id
            ###########################
            ## semantic segmentation ##
            ###########################
            # cur_voxel_semantic = cur_voxel_prediction.cpu().detach().numpy().astype(np.int32)  # (17,480,360,32)
            # cur_voxel_semantic = np.argmax(cur_voxel_semantic, axis=0)  # (480, 360, 32)
            # sem_output = cur_voxel_semantic[point_x_index, point_y_index, point_z_index]
            # sem_target = gt_points_label[cur_sample_index]
            # sem_hist_list.append(fast_hist_crop(output=sem_output, target=sem_target, unique_label=self.unique_label))

        panoptic_outputs = {
            'voxel_pred_panoptic_list': voxel_panoptic_results,  # voxel-wise panoptic segmentation prediction
            'point_pred_panoptic_list': point_panoptic_results,  # point-wise panoptic segmentation prediction
            'point_gt_label_list': point_gt_labels_list,  # point-wise semantic label
            'point_gt_instance_list': point_gt_instances_list,  # point-wise instance label
            # 'sem_hist_list': sem_hist_list
        }
        output_dict.update(panoptic_outputs)
        return output_dict

    def get_training_loss(self, output_dict, batch_dict):
        # predictions of network
        pred_voxel_sem = output_dict['voxel_prediction']  # (B, 20, 480, 360, 32)
        pred_center = output_dict['pred_center']  # (B, 1, 480, 360)
        pred_offset = output_dict['pred_offset']  # (B, 2, 480, 360)

        # ground truth
        gt_voxel_sem = batch_dict['voxel_sem_label']  # (B, 480, 360, 32)
        gt_center = batch_dict['gt_center']  # (B, 1, 480, 360)
        gt_offset = batch_dict['gt_offset']  # (B, 2, 480, 360)

        total_loss, loss_dict = self.cal_panoptic_loss(
            sem_prediction=pred_voxel_sem,
            center_prediction=pred_center,
            offset_prediction=pred_offset,
            voxel_label=gt_voxel_sem,
            center_gt=gt_center,
            offset_gt=gt_offset
        )

        return total_loss, loss_dict

    def cal_panoptic_loss(self, sem_prediction, center_prediction, offset_prediction, voxel_label, center_gt,
                          offset_gt):
        loss_dict = {}

        ce_loss = self.ce_loss_fn(sem_prediction, voxel_label)
        lz_loss = lovasz_softmax(
            torch.nn.functional.softmax(sem_prediction, dim=1), voxel_label, ignore=self.ignore_label)
        total_loss = lz_loss + ce_loss

        # center heatmap loss
        center_mask = (center_gt > 0.01) | (torch.min(torch.unsqueeze(voxel_label, 1), dim=4)[0] < self.ignore_label)
        # view = center_gt.cpu().numpy()
        center_loss = self.center_loss_fn(center_prediction, center_gt) * center_mask
        # safe division
        if center_mask.sum() > 0:
            center_loss = center_loss.sum() / center_mask.sum() * self.center_loss_weight
        else:
            center_loss = center_loss.sum() * 0

        total_loss += center_loss
        ################################################################
        ########################## offset loss #########################
        offset_mask = offset_gt != 0
        offset_loss = self.offset_loss_fn(offset_prediction, offset_gt) * offset_mask
        # safe division
        if offset_mask.sum() > 0:
            offset_loss = offset_loss.sum() / offset_mask.sum() * self.offset_loss_weight
        else:
            offset_loss = offset_loss.sum() * 0
        ############################################
        loss_dict['InstanceOffsetLoss'] = offset_loss.cpu().detach().item()
        loss_dict['LovaszSoftmaxLoss'] = lz_loss.cpu().detach().item()
        loss_dict['CrossEntropyLoss'] = ce_loss.cpu().detach().item()
        loss_dict['InstanceCenterHeatmapLoss'] = center_loss.cpu().detach().item()

        total_loss += offset_loss
        loss_dict['loss'] = total_loss.cpu().detach().item()

        return total_loss, loss_dict

    def training_step(self, batch, batch_idx):
        if self._config["freeze_layers"]:
            self.cylinder_3d_generator.eval()
        else:
            self.cylinder_3d_generator.train()
        self.cylinder_3d_spconv_seg.train()
        self.UNet.train()
        ########################################################
        batch_size = len(batch['pc_features'])
        batch['batch_size'] = batch_size
        if self.model_point_name == 'minkunet':
            raise Exception("Not implementation")
        elif self.model_point_name == 'voxelnet':
            raise Exception("Not implementation")
        elif self.model_point_name == 'cylinder3d':
            total_loss, loss_dict = self.forward_network(batch)
            ce_loss = loss_dict['CrossEntropyLoss']
            lovasz_loss = loss_dict['LovaszSoftmaxLoss']
            offset_loss = loss_dict['InstanceOffsetLoss']
            heatmap_loss = loss_dict['InstanceCenterHeatmapLoss']
        else:
            raise Exception("Unknow 3d backbone")

        self.log_dict(
            {
                "ls": total_loss,
                "ce_ls": ce_loss,
                "lov_ls": lovasz_loss,
                "off_ls": offset_loss,
                "hm_ls": heatmap_loss,
            },
            on_step=True, on_epoch=False, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True
        )
        self.train_losses.append(total_loss.detach().cpu())
        return total_loss

    def on_train_epoch_end(self) -> None:
        self.epoch += 1

    def on_validation_epoch_start(self) -> None:
        # 用于评估
        self.point_sem_pred_list = []
        self.point_inst_pred_list = []
        self.point_sem_gt_list = []
        self.point_inst_gt_list = []
        # self.sem_hist_list = []

    def validation_step(self, batch, batch_idx):
        if self.model_point_name == 'cylinder3d':
            self.cylinder_3d_generator.eval(),
            self.cylinder_3d_spconv_seg.eval(),
            self.UNet.eval(),
        else:
            raise Exception("Unsupported 3d model")
        torch.cuda.empty_cache()
        pred_dict = self.forward_network(batch)

        point_pred_panoptic_list = pred_dict['point_pred_panoptic_list']
        point_gt_label_list = pred_dict['point_gt_label_list']
        point_gt_instance_list = pred_dict['point_gt_instance_list']

        for batch_i in range(len(point_pred_panoptic_list)):
            sample_point_pred_panoptic = point_pred_panoptic_list[batch_i]
            sample_point_gt_label = point_gt_label_list[batch_i]
            sample_point_gt_instance = point_gt_instance_list[batch_i]

            point_sem_pred, point_inst_pred = sample_point_pred_panoptic & 0xFFFF, sample_point_pred_panoptic
            point_sem_gt, point_inst_gt = sample_point_gt_label, sample_point_gt_instance

            # 注意：需要将Tensor从GPU移动到CPU并且不再跟踪其梯度。
            self.point_sem_pred_list.append(point_sem_pred)
            self.point_inst_pred_list.append(point_inst_pred)
            self.point_sem_gt_list.append(point_sem_gt)
            self.point_inst_gt_list.append(point_inst_gt)

    def on_validation_epoch_end(self, ) -> None:
        rank = self.global_rank
        world_size = self.trainer.num_devices
        ######################################
        if self.trainer.num_devices > 1:
            # 保存文件然后读取
            save_dict = {
                "point_sem_pred": self.point_sem_pred_list,  # []
                "point_inst_pred": self.point_inst_pred_list,  # []
                "point_sem_gt": self.point_sem_gt_list,  # []
                "point_inst_gt": self.point_inst_gt_list,  # []
            }
            with open(os.path.join(self.temp_dir, 'result_part_{}_{}.pkl'.format(rank, world_size)), 'wb') as f:
                pickle.dump(save_dict, f)
            self.trainer.strategy.barrier()
            point_sem_pred_list, point_inst_pred_list, point_sem_gt_list, point_inst_gt_list = [], [], [], []
            for i in range(world_size):
                part_file = os.path.join(self.temp_dir, 'result_part_{}_{}.pkl'.format(i, world_size))
                cur_dict = pickle.load(open(part_file, 'rb'))

                cur_point_sem_pred = cur_dict['point_sem_pred']  # [ np.array,...,np.array]
                cur_point_inst_pred = cur_dict['point_inst_pred']  # [ np.array,...,np.array]
                cur_point_sem_gt = cur_dict['point_sem_gt']  # [ np.array,...,np.array]
                cur_point_inst_gt = cur_dict['point_inst_gt']  # [ np.array,...,np.array]

                point_sem_pred_list.extend(cur_point_sem_pred)
                point_inst_pred_list.extend(cur_point_inst_pred)
                point_sem_gt_list.extend(cur_point_sem_gt)
                point_inst_gt_list.extend(cur_point_inst_gt)

        else:
            point_sem_pred_list = self.point_sem_pred_list
            point_inst_pred_list = self.point_inst_pred_list
            point_sem_gt_list = self.point_sem_gt_list
            point_inst_gt_list = self.point_inst_gt_list

        #########################
        for index in range(len(point_sem_pred_list)):
            self.panoptic_evaluator.addBatchPanoptic(
                x_sem_row=point_sem_pred_list[index],
                x_inst_row=point_inst_pred_list[index],
                y_sem_row=point_sem_gt_list[index],
                y_inst_row=point_inst_gt_list[index]
            )

        class_PQ, class_SQ, class_RQ, class_all_PQ, class_all_SQ, class_all_RQ = self.panoptic_evaluator.getPQ()
        # miou, ious = self.panoptic_evaluator.getSemIoU()
        best_epoch = False
        if self.best_PQ < class_PQ:
            self.best_PQ = class_PQ
            best_epoch = True

        results_dict = {
            'all_class_pq': class_all_PQ.tolist(),
            'all_class_sq': class_all_SQ.tolist(),
            'all_class_rq': class_all_RQ.tolist(),
            'pq': class_PQ,
            'sq': class_SQ,
            'rq': class_RQ,
            'epoch': self.current_epoch,
            'best_PQ': self.best_PQ
        }

        # 保存权重和结果
        if best_epoch:
            self.save(results_dict, save_weights=True, best=True)

        self.log_dict(
            {
                "pq": class_PQ,
                "sq": class_SQ,
                "rq": class_RQ,
                "best_PQ": self.best_PQ,
            },
            on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        # 仅保存json文件
        self.save(results_dict, save_weights=False, best=False)

        self.point_sem_pred_list = []
        self.point_inst_pred_list = []
        self.point_sem_gt_list = []
        self.point_inst_gt_list = []

    @rank_zero_only
    def save(self, results_dict, save_weights=True, best=False):
        if best:
            weight_path = os.path.join(self.working_dir, f"best_model.pt")
            json_path = os.path.join(self.working_dir, "best_results.json")
        else:
            weight_path = os.path.join(self.working_dir, f"model_{self.current_epoch}.pt")
            json_path = os.path.join(self.working_dir, f"results_{self.current_epoch}.json")

        results_json = json.dumps(results_dict, sort_keys=True)
        with open(json_path, 'w') as f:
            f.write(results_json)

        if save_weights:
            torch.save(
                {
                    "cylinder_3d_generator": self.cylinder_3d_generator.state_dict(),
                    "cylinder_3d_spconv_seg": self.cylinder_3d_spconv_seg.state_dict(),
                    "unet": self.UNet.state_dict(),
                    "config": self._config
                },
                weight_path
            )
