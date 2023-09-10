import torch
from torch import nn
import math
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np

class MemoryUnit(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim #memitems的数量
        self.fea_dim = fea_dim #编码后特征的维度
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # M * F
        self.bias = None
        self.shrink_thres= shrink_thres
        self.max_index=[]
        
        # self.hard_sparse_shrink_opt = nn.Hardshrink(lambd=shrink_thres)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self,input): #input T*F
        att_weight = F.linear(input, self.weight) #计算相似度矩阵 (T*F)*(F*M)=T*M
        att_weight = F.softmax(att_weight, dim=1) #attention T*M 在M上相加为1
        if (self.shrink_thres > 0): #是否有阈值，有的话进行权重的稀疏化处理，减少信息冗杂
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1) #归一化操作
        self.max_index+=torch.argmax(att_weight,dim=1).view(1,-1).tolist()[0]
        mem_trans = self.weight.permute(1, 0) #weight转置 M*F 记忆存储模块
        output = F.linear(att_weight, mem_trans) #潜在表示
        return {'output': output, 'att': att_weight}

    def extra_repr(self):
        return 'mem_dim={}, fea_dim={}'.format(
            self.mem_dim, self.fea_dim is not None
        )
class MemModule(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025, device='cuda'):
        super(MemModule, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.shrink_thres) #记忆存储模块
        self.max_index=self.memory.max_index
    def forward(self, input):
        s = input.data.shape
        x = input.contiguous()
        x = x.view(-1, s[1])
        y_and = self.memory(x)
        y = y_and['output']
        att = y_and['att']
        
        return {'output': y, 'att': att,'max_index':self.max_index,'mem_fea':self.memory.weight,'fea_list':x}

#紧缩处理稀疏数据，也就是权重的稀疏化处理
def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output