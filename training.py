'''
共有8个数据集，所以要注意程序的泛用性
具体实现分为以下几个步骤
step1：数据的预处理，分训练集和测试集
step2：编码器的训练
step3：mem存储模块的形成，也是训练的过程
step4：test并保存评价指标
注意：
1.两个训练的模型注意保存，花费时间太多（.h5）
2.中间比重最大的index记得进行保存
3.利用80%的数据进行训练，20%数据用于测试，
'''


import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import dataset
from options import TrainOptions
import utils
from models import AutoEncoderMem
from models import EntropyLossEncap

#解析器加载参数
opt_parser = TrainOptions()
opt = opt_parser.parse(is_print=True)
use_cuda = opt.UseCUDA
device = torch.device("cuda" if use_cuda else "cpu")

##种子及算法设置
utils.seed(opt.Seed)
if(opt.IsDeter):
    torch.backends.cudnn.benchmark = False #计算性能相关，关闭节省空间
    torch.backends.cudnn.deterministic = True #确定性算法，保证相同条件结果一样

##获取model的设置
model_setting = utils.get_model_setting(opt)
print('Setting: %s' % (model_setting))

##数据及模型参数
batch_size_in = opt.BatchSize
learning_rate = opt.LR
max_epoch_num = opt.EpochNum

f_dim_in = opt.FeatureDim   #特征的维数
chnum_in_ = opt.ImgChnNum      # 图像的channel数 在此无用
framenum_in_ = opt.FrameNum  # 视频帧数 在此无用
mem_dim_in = opt.MemDim  # memitems的数量
entropy_loss_weight = opt.EntropyLossWeight #熵的权重
sparse_shrink_thres = opt.ShrinkThres #阈值大小

img_crop_size = 0

print('bs=%d, lr=%f, entrloss=%f, shr=%f, memdim=%d' % (batch_size_in, learning_rate, entropy_loss_weight, sparse_shrink_thres, mem_dim_in))

#路径设置
##数据路径，需要修改
file_num=opt.Filenum
file_list = ['arrhythmia_normalization.csv', 'cardio_normalization.csv', 'fraud_normalization.csv',
                 'mammography_normalization.csv', 'nslkdd_normalization.csv', 'satellite_normalization.csv',
                 'shuttle_normalization.csv', 'spambase_normalization.csv']
data_root = opt.DataRoot + 'train' + '/train_'
file_path = data_root + file_list[file_num]

##models保存路径
saving_root = opt.ModelRoot
saving_model_path = os.path.join(saving_root, 'model_' + model_setting + '/')
utils.mkdir(saving_model_path)


#加载data
csv_dataset = utils.CsvDataset(filepath=file_path)
csv_dataloder = DataLoader(csv_dataset,
                           batch_size=batch_size_in,
                           shuffle=True,
                           num_workers=opt.NumWorker
                           )
###model
if(opt.ModelName=='MemAE'):
    model = AutoEncoderMem(f_dim_in, mem_dim_in, shrink_thres=sparse_shrink_thres)
else:
    model = []
    print('Wrong model name.')
model.apply(utils.weights_init)

#####
device = torch.device("cuda" if use_cuda else "cpu")
model.to(device)
tr_recon_loss_func = nn.MSELoss().to(device)
tr_entropy_loss_func = EntropyLossEncap().to(device)
tr_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

##
data_loader_len = len(csv_dataloder)
textlog_interval = opt.TextLogInterval
snap_save_interval = opt.SnapInterval
save_check_interval = opt.SaveCheckInterval
global_ite_idx = 0 # for logging
for epoch_idx in range(0, max_epoch_num):
    for batch_idx, (batch_x, batch_y) in enumerate(csv_dataloder):
        batch_x=batch_x.to(device)
        if (opt.ModelName == 'MemAE'):
            recon_res = model(batch_x)
            recon_frames = recon_res['output']
            att_w = recon_res['att']
            max_index=recon_res['max_index']
            loss = tr_recon_loss_func(recon_frames, batch_x) #重构误差
            recon_loss_val = loss.item()
            entropy_loss = tr_entropy_loss_func(att_w)
            entropy_loss_val = entropy_loss.item()
            loss = loss + entropy_loss_weight * entropy_loss
            loss_val = loss.item()

            #优化器
            tr_optimizer.zero_grad()
            loss.backward()
            tr_optimizer.step()
            ##
        if ((batch_idx % textlog_interval) == 0):
            print('[%s, epoch %d/%d, bt %d/%d] loss=%f, rc_losss=%f, ent_loss=%f' % (
            model_setting, epoch_idx, max_epoch_num, batch_idx, data_loader_len, loss_val, recon_loss_val,
            entropy_loss_val))
        if ((global_ite_idx % snap_save_interval) == 0):
            torch.save(model.state_dict(), '%s/%s_snap.pt' % (saving_model_path, model_setting))
        global_ite_idx += 1
    if ((epoch_idx % save_check_interval) == 0):
        torch.save(model.state_dict(), '%s/%s_epoch_%04d.pt' % (saving_model_path, model_setting, epoch_idx))
#print(max_index)
#print(len(max_index))
torch.save(model.state_dict(), '%s/%s_epoch_%04d_final.pt' % (saving_model_path, model_setting, epoch_idx))

