import torch
from torch import nn


class NCELoss(nn.Module):
    """
    Compute the PointInfoNCE loss
    """

    def __init__(self, temperature):
        super(NCELoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, k, q, verbose=False):
        if verbose:
            print(k)
            print(q)
        logits = torch.mm(k, q.transpose(1, 0))  # 向量积; k点特征，q图像特征  k: [4553,64] . q: [4553,64] ; logits: [4553,4553]
        target = torch.arange(k.shape[0], device=k.device).long()
        out = torch.div(logits, self.temperature)
        out = out.contiguous()  #

        loss = self.criterion(out, target)
        return loss
