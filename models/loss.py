import torch
from torch import nn

def feature_map_permute(input):
    #保留图像变换的函数，其实在这里排不上用场
    #可以不用，但不能没有
    s = input.data.shape
    x = input.contiguous()
    x = x.view(-1, s[1])
    return x

class EntropyLoss(nn.Module):
    def __init__(self, eps = 1e-12):
        super(EntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, x):
        b = x * torch.log(x + self.eps)
        b = -1.0 * b.sum(dim=1)
        b = b.mean()
        return b

class EntropyLossEncap(nn.Module):
    def __init__(self, eps = 1e-12):
        super(EntropyLossEncap, self).__init__()
        self.eps = eps
        self.entropy_loss = EntropyLoss(eps)

    def forward(self, input):
        score = feature_map_permute(input)
        ent_loss_val = self.entropy_loss(score)
        return ent_loss_val
