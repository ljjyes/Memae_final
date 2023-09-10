import numpy as np
import torch
import random
import os
def seed(seed_val): #所有包设置随机种子
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    random.seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)

def get_model_setting(opt):
    if(opt.ModelName == 'MemAE'):
        model_setting = opt.ModelName + '_' + opt.ModelSetting + '_' + opt.Dataset + '_MemDim' + str(opt.MemDim) \
                        + '_EntW' + str(opt.EntropyLossWeight) + '_ShrThres' + str(opt.ShrinkThres) \
                        + '_Seed' + str(opt.Seed) + '_' + opt.Suffix
    elif(opt.ModelName == 'AE'):
        model_setting = opt.ModelName + '_' + opt.ModelSetting + '_' + opt.Dataset \
                        + '_' + opt.Suffix
    else:
        model_setting = ''
        print('Wrong Model Name.')
    return model_setting
def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)