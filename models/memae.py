import torch
from torch import nn

from .memory_module import MemModule

class AutoEncoderMem(nn.Module):
    def __init__(self, chnum_in, mem_dim, shrink_thres=0.0025):
        super(AutoEncoderMem, self).__init__()
        print('AutoEncoderMem')
        self.chnum_in = chnum_in
        f_num = 28
        f_num1 = 14
        f_num2= 7
        f_num3= 3
        self.encoder=nn.Sequential(
            nn.Linear(self.chnum_in,f_num),
            nn.ReLU(),
            #Tanh
            nn.Linear(f_num,f_num1),
            nn.ReLU(),
            nn.Linear(f_num1, f_num2),
            nn.ReLU(),
            nn.Linear(f_num2, f_num3),
            nn.ReLU()
        )
        self.mem_rep = MemModule(mem_dim=mem_dim, fea_dim=f_num3, shrink_thres=shrink_thres)
        self.decoder=nn.Sequential(
            nn.Linear(f_num3,f_num2),
            nn.ReLU(),
            nn.Linear(f_num2, f_num1),
            nn.ReLU(),
            nn.Linear(f_num1, f_num),
            nn.ReLU(),
            nn.Linear(f_num,self.chnum_in)
        )

    def forward(self, x):
        f = self.encoder(x)
        res_mem = self.mem_rep(f)
        f = res_mem['output']
        att = res_mem['att']
        max_index=res_mem['max_index']
        output = self.decoder(f)
        return {'output': output, 'att': att,'max_index':max_index}
