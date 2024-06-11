# -*-coding:utf-8-*-
"""
    Author: chenhaomingbob
    E-mail: chenhaomingbob@163.com
    Time: 2022/07/03
    Description: 
"""
import torch
import torch.nn.functional as F
import torch_scatter


def find_instance_center(ctr_hmp, threshold=0.1, nms_kernel=5, top_k=None, polar=False):
    """
    Find the center points from the center heatmap.
    Arguments:
        ctr_hmp: A Tensor of shape [N, 1, H, W] of raw center heatmap output, where N is the batch size,
            for consistent, we only support N=1.
        threshold: A Float, threshold applied to center heatmap score.
        nms_kernel: An Integer, NMS max pooling kernel size.
        top_k: An Integer, top k centers to keep.
    Returns:
        A Tensor of shape [K, 2] where K is the number of center points. The order of second dim is (y, x).
    """
    if ctr_hmp.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')

    # thresholding, setting values below threshold to -1
    ctr_hmp = F.threshold(ctr_hmp, threshold, -1)

    # NMS
    if polar:
        # 极坐标的情况下，是个圈。mode要设置为circular!!。并且至增加 360那个维度，不增加480那个
        nms_padding = (nms_kernel - 1) // 2  # nms_padding = 2
        ctr_hmp_max_pooled = F.pad(ctr_hmp, (nms_padding, nms_padding, 0, 0),
                                   mode='circular')  # [1, 1, 480, 360] -> [1, 1, 480, 364]
        ctr_hmp_max_pooled = F.max_pool2d(ctr_hmp_max_pooled, kernel_size=nms_kernel, stride=1,
                                          padding=(nms_padding, 0))  # [1, 1, 480, 364] -> [1, 1, 480, 360]
    else:
        nms_padding = (nms_kernel - 1) // 2
        ctr_hmp_max_pooled = F.max_pool2d(ctr_hmp, kernel_size=nms_kernel, stride=1, padding=nms_padding)
    ctr_hmp[ctr_hmp != ctr_hmp_max_pooled] = -1

    # squeeze first two dimensions
    ctr_hmp = ctr_hmp.squeeze()
    assert len(ctr_hmp.size()) == 2, 'Something is wrong with center heatmap dimension.'

    # find non-zero elements
    ctr_all = torch.nonzero(ctr_hmp > 0)
    if top_k is None:
        return ctr_all
    elif ctr_all.size(0) < top_k:
        return ctr_all
    else:
        # find top k centers.
        top_k_scores, _ = torch.topk(torch.flatten(ctr_hmp), top_k)
        return torch.nonzero(ctr_hmp > top_k_scores[-1])


def group_pixels(ctr, offsets, polar=False):
    """
    Gives each pixel in the image an instance id.
    Arguments:
        ctr: A Tensor of shape [K, 2] where K is the number of center points. The order of second dim is (y, x).
        offsets: A Tensor of shape [N, 2, H, W] of raw offset output, where N is the batch size,
            for consistent, we only support N=1. The order of second dim is (offset_y, offset_x).
    Returns:
        A Tensor of shape [1, H, W] (to be gathered by distributed data parallel).
    """
    if offsets.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')

    offsets = offsets.squeeze(0)
    height, width = offsets.size()[1:]

    # generates a coordinate map, where each location is the coordinate of that loc
    y_coord = torch.arange(height, dtype=offsets.dtype, device=offsets.device).repeat(1, width, 1).transpose(1, 2)
    x_coord = torch.arange(width, dtype=offsets.dtype, device=offsets.device).repeat(1, height, 1)
    coord = torch.cat((y_coord, x_coord), dim=0)

    ctr_loc = coord + offsets
    ctr_loc = ctr_loc.reshape((2, height * width)).transpose(1, 0)

    # ctr: [K, 2] -> [K, 1, 2]
    # ctr_loc = [H*W, 2] -> [1, H*W, 2]
    ctr = ctr.unsqueeze(1)
    ctr_loc = ctr_loc.unsqueeze(0)

    # distance: [K, H*W]
    distance = ctr - ctr_loc
    if polar:
        # torch.fmod(x,y): 返回x/y的浮点余数
        distance[:, :, 0] = torch.add(torch.fmod(torch.add(distance[:, :, 0],
                                                           width / 2),
                                                 width),
                                      -width / 2)
    distance = torch.norm(distance, dim=-1)

    # finds center with minimum distance at each location, offset by 1, to reserve id=0 for stuff
    instance_id = torch.argmin(distance, dim=0).reshape((1, height, width)) + 1  # (1,480,360)
    return instance_id


def get_instance_segmentation(sem_seg, ctr_hmp, offsets, thing_list, threshold=0.1, nms_kernel=5, top_k=None,
                              thing_seg=None, polar=False):
    """
    Post-processing for instance segmentation, gets class agnostic instance id map.
    Arguments:
        sem_seg: A Tensor of shape [1, H, W, Z], predicted semantic label.
        ctr_hmp: A Tensor of shape [N, 1, H, W] of raw center heatmap output, where N is the batch size,
            for consistent, we only support N=1.
        offsets: A Tensor of shape [N, 2, H, W] of raw offset output, where N is the batch size,
            for consistent, we only support N=1. The order of second dim is (offset_y, offset_x).
        thing_list: A List of thing class id.
        threshold: A Float, threshold applied to center heatmap score.
        nms_kernel: An Integer, NMS max pooling kernel size.
        top_k: An Integer, top k centers to keep.
        thing_seg: A Tensor of shape [1, H, W, Z], predicted foreground mask, if not provided, inference from
            semantic prediction.
    Returns:
        A Tensor of shape [1, H, W] (to be gathered by distributed data parallel).
        A Tensor of shape [1, K, 2] where K is the number of center points. The order of second dim is (y, x).
    """
    # if thing_seg is None:
    #     # gets foreground segmentation
    #     thing_seg = torch.zeros_like(sem_seg)
    #     for thing_class in thing_list:
    #         thing_seg[sem_seg == thing_class] = 1
    # if thing_seg.dim() == 4:
    #     # [1, H, W, Z] --> [1, H, W]
    #     thing_seg = torch.max(thing_seg,dim=3)

    # 找实例的中心  [N_instance,2]  ctr包含所有实例中心点，以及中心点的xy坐标（BEV视角下的）
    ctr = find_instance_center(ctr_hmp, threshold=threshold, nms_kernel=nms_kernel, top_k=top_k, polar=polar)
    if ctr.size(0) == 0:
        return torch.zeros_like(thing_seg[:, :, :, 0]), ctr.unsqueeze(0)
    # ins_seg 为instance map, 其shape跟BEV图一样，每个像素上包含了代表instance_id的值。且instance_id从1开始算起。
    ins_seg = group_pixels(ctr, offsets, polar=polar)  # [1,480,360]
    batch_ctr = ctr.unsqueeze(0)  # [N_instance,2] -> [1, N_instance, 2]
    return ins_seg, batch_ctr


def merge_semantic_and_instance(sem_seg, sem, ins_seg, label_divisor, thing_list, void_label, thing_seg):
    """
    Post-processing for panoptic segmentation, by merging semantic segmentation label and class agnostic
        instance segmentation label.
    Arguments:
        sem_seg: A Tensor of shape [1, H, W, Z], predicted semantic label.
        sem: A Tensor of shape [1, C, H, W, Z], predicted semantic logit.
        ins_seg: A Tensor of shape [1, H, W], predicted instance label.
        label_divisor: An Integer, used to convert panoptic id = semantic id * label_divisor + instance_id.
        thing_list: A List of thing class id.
        void_label: An Integer, indicates the region has no confident prediction.
        thing_seg: A Tensor of shape [1, H, W, Z], predicted foreground mask.
    Returns:
        A Tensor of shape [1, H, W, Z] (to be gathered by distributed data parallel).
    Raises:
        ValueError, if batch size is not 1.
    """
    # try to avoid the for loop
    # semantic_thing_seg:[1,480,360,32], 网络会预测处每个voxel的label。这个semantic_thing_seg的作用类似mask，用于找出属于thing class的voxel
    semantic_thing_seg = sem_seg <= max(thing_list)
    # [1,480,360] -> [1,480,360,1] -> [1,480,360,32] 此处有一个假设：即处于相同xy坐标(笛卡尔坐标系)/ \theta r坐标（极坐标系）的voxel，有着相同的instance id.
    ins_seg = torch.unsqueeze(ins_seg, 3).expand_as(sem_seg)  # (1,480,360) -> (1,480,360,32)
    # 生成最终thing mask.
    # 如果voxel属于thing，当且仅当 该voxel的instance id >0 (e.g. ins_seg>0) & 根据网络的预测，该voxel的class属于thing class （e.g. semantic_thing_seg） & 根据gt的标注，该voxel的class属于thing class (e.g. thing seg)
    # 注: 在验证集上, 由于有标注信息，thing_seg是存在的；在测试集上，thing_seg为None
    if thing_seg is not None:
        thing_mask = (ins_seg > 0) & semantic_thing_seg & thing_seg
    else:
        thing_mask = (ins_seg > 0) & semantic_thing_seg

    if not torch.nonzero(thing_mask).size(0) == 0:
        # 整个voxel中thing的数量不为0，即整个voxel中存在thing
        # sem.permute(0, 2, 3, 4, 1)[thing_mask]: (Num_thing, 19)   ins_seg[thing_mask]: (N_thing)
        # sem_sum: (N_insatnce, 19)  对于属于同一个instance id的voxel们，将他们的特征向量进行相加。
        sem_sum = torch_scatter.scatter_add(sem.permute(0, 2, 3, 4, 1)[thing_mask], ins_seg[thing_mask], dim=0)
        # 通过切片的方式（sem_sum[:,:max(thing_list)）取出特征向量中属于thing class的一些维度，然后在这些thing 维度上通过argmax来获取最有可能的class
        class_id = torch.argmax(sem_sum[:, :max(thing_list)], dim=1)  # (N_instance,)
        # 将instance_id和class_id进行合并。具体来说，每个voxel元素上都有的一个复合值（type为int64），该复合值的高位表示instance_id，复合值的低位表示class_id
        sem_seg[thing_mask] = (ins_seg[thing_mask] * label_divisor) + class_id[
            ins_seg[thing_mask]] + 1  # 这边的+1是配合class_id用的
        # torch.unique(sem_seg)
        # torch.unique(sem_seg&0xFFFF)
    else:
        # 整个voxel中thing的数量为0
        sem_seg[semantic_thing_seg & thing_seg] = void_label
    return sem_seg


def get_panoptic_segmentation(sem, ctr_hmp, offsets, thing_list, label_divisor=2 ** 16, void_label=0,
                              threshold=0.1, nms_kernel=5, top_k=100, foreground_mask=None, polar=False):
    """
    Post-processing for panoptic segmentation.
    Arguments:
        sem: A Tensor of shape [N, C, H, W, Z] of raw semantic output, where N is the batch size, for consistent,
            we only support N=1.
        ctr_hmp: A Tensor of shape [N, 1, H, W] of raw center heatmap output, where N is the batch size,
            for consistent, we only support N=1.
        offsets: A Tensor of shape [N, 2, H, W] of raw offset output, where N is the batch size,
            for consistent, we only support N=1. The order of second dim is (offset_y, offset_x).
        thing_list: A List of thing class id.
        label_divisor: An Integer, used to convert panoptic id = instance_id * label_divisor + semantic_id.
        void_label: An Integer, indicates the region has no confident prediction.
        threshold: A Float, threshold applied to center heatmap score.
        nms_kernel: An Integer, NMS max pooling kernel size.
        top_k: An Integer, top k centers to keep.
        foreground_mask: A processed Tensor of shape [N, H, W, Z], we only support N=1.
    Returns:
        A Tensor of shape [1, H, W, Z] (to be gathered by distributed data parallel), int64.
    Raises:
        ValueError, if batch size is not 1.
    """
    if sem.dim() != 5 and sem.dim() != 4:
        raise ValueError('Semantic prediction with un-supported dimension: {}.'.format(sem.dim()))
    if sem.dim() == 5 and sem.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')
    if ctr_hmp.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')
    if offsets.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')
    if foreground_mask is not None:
        if foreground_mask.dim() != 4:
            raise ValueError('Foreground prediction with un-supported dimension: {}.'.format(sem.dim()))

    if sem.dim() == 5:
        # (B,20,480,360,32) -> (B,480,360,32) dim=1上的每个通道代表一个class，通过argmax在dim=1上取值最大的通道index
        semantic = torch.argmax(sem, dim=1)  # [0~19]
        # shift back to original label idx
        semantic = torch.add(semantic, 1)  # [0~19] -> [1~20]
        sem = F.softmax(sem, dim=1)  # (B,20,480,360,32) -> (B,20,480,360,32)
    else:
        semantic = sem.type(torch.ByteTensor).cuda()
        # shift back to original label idx
        semantic = torch.add(semantic, 1).type(torch.LongTensor).cuda()  # [0~19,255] -> [0~20] (N,1)
        one_hot = torch.zeros((sem.size(0), torch.max(semantic).item() + 1,
                               sem.size(1), sem.size(2), sem.size(3))).cuda()
        # one_hot = torch.zeros((sem.size(0), torch.max(semantic).item() + 1, sem.size(1), sem.size(2), sem.size(3))).cuda()
        sem = one_hot.scatter_(dim=1, index=torch.unsqueeze(semantic, 1), src=1.)  # (1, 21, 480, 360, 32)
        sem = sem[:, 1:, :, :, :]  # (1, 20, 480, 360, 32)  去掉第0个channel (unlabeled point)

    # foreground_mask 用于区分 thing class 和stuff class
    if foreground_mask is not None:
        thing_seg = foreground_mask
    else:
        thing_seg = None
    # ins_seg: [1,480,360] 为instance map, map上每个像素的值代表该像素对应的instance_id.  center: [1,N_instance,2] 实例的中心点，具体为中心点的xy坐标。
    instance, center = get_instance_segmentation(semantic, ctr_hmp, offsets, thing_list,
                                                 threshold=threshold, nms_kernel=nms_kernel, top_k=top_k,
                                                 thing_seg=thing_seg, polar=polar)
    # instance: (1, 480, 360)  center: (1,num_center,2) 56
    panoptic = merge_semantic_and_instance(semantic, sem, instance, label_divisor, thing_list, void_label, thing_seg)

    return panoptic, center
