from torch import nn

class feature_coder(nn.Module):
    def __init__(self,f_dim):
        super(feature_coder, self).__init__()
        f_num=128
        f_num1=64
        self.encoder=nn.Sequential(
            nn.Linear(f_dim,f_num),
            nn.ReLU(),
            nn.Linear(f_num,f_num1),
            nn.ReLU()
        )
        self.decoder=nn.Sequential(
            nn.Linear(f_num1,f_num),
            nn.ReLU(),
            nn.Linear(f_num,f_dim),
            nn.ReLU()
        )
    def forward(self,x):
        f = self.encoder(x)
        out = self.decoder(f)
        return out
