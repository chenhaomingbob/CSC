import torch


def convert_spid(superpixels_type, sp_id):
    """

        return: 需要保证返回的sp_id,是从0开始计算的
    """
    # return:
    if superpixels_type == "dinov2_ade20k":
        # dinov2_ade20k的superpixel id最小值为0
        valid_sp_id = sp_id - 1
    else:
        # slic的sp_id最小值为0,且是有意义的
        # seem的sp_id最小值为0,但0是无意义的
        valid_sp_id = sp_id

    return valid_sp_id
