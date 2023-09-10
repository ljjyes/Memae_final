import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import dataset
import scipy.io as sio
from options.testing_options import TestOptions
import utils
import time
from models import  AutoEncoderMem,feature_coder

opt_parser = TestOptions()
opt = opt_parser.parse(is_print=True)
use_cuda = opt.UseCUDA
device = torch.device("cuda" if use_cuda else "cpu")

batch_size_in = opt.BatchSize
chnum_in_ = opt.ImgChnNum
f_dim_in = opt.FeatureDim
framenum_in_ = opt.FrameNum
mem_dim_in = opt.MemDim
sparse_shrink_thres = opt.ShrinkThres


model_setting = utils.get_model_setting(opt)

file_num=opt.Filenum
file_list = ['arrhythmia_normalization', 'cardio_normalization', 'fraud_normalization',
                 'mammography_normalization', 'nslkdd_normalization', 'satellite_normalization',
                 'shuttle_normalization', 'spambase_normalization']
data_root = opt.DataRoot  +'/test'+ '/test_'
#data_root = opt.DataRoot  +'/train'+ '/train_'
file_path = data_root + file_list[file_num] + '.csv'

model_root = opt.ModelRoot
if(opt.ModelFilePath):
    model_path = opt.ModelFilePath
else:
    model_path = os.path.join(model_root+model_setting+'/' , model_setting + opt.Suffix1 + '.pt')

te_res_root = opt.OutRoot
te_res_path = te_res_root + '/' + 'res_' + model_setting
utils.mkdir(te_res_path)

###### loading trained model
if (opt.ModelName == 'AE'):
    model = feature_coder(f_dim_in)
elif(opt.ModelName=='MemAE'):
    model = AutoEncoderMem(f_dim_in, mem_dim_in, shrink_thres=sparse_shrink_thres)
else:
    model = []
    print('Wrong Name.')

model_para = torch.load(model_path)
model.load_state_dict(model_para)
model.to(device)
model.eval()

with torch.no_grad():
    # 加载data
    csv_dataset = utils.CsvDataset(filepath=file_path)
    csv_dataloder = DataLoader(csv_dataset,
                               batch_size=batch_size_in,
                               shuffle=False
                               )
    recon_error_list=[]
    max_index_list=[]
    fea_list=[]
    for batch_idx, (batch_x, batch_y) in enumerate(csv_dataloder):
        batch_x = batch_x.to(device)
        if (opt.ModelName == 'AE'):
            recon_res = model(batch_x)
            recon_np=recon_res.data
            input_np=batch_x.data
            r=recon_np-input_np
            recon_error = np.mean(r ** 2)#**0.5
            recon_error_list += [recon_error]
        elif(opt.ModelName == 'MemAE'):
            recon_res = model(batch_x)
            recon_frames = recon_res['output']
            fea_list+=recon_res['fea_list']
            r = recon_frames - batch_x
            sp_error_map = torch.sum(r ** 2, dim=1) ** 0.5
            recon_error_list += [sp_error_map]
        else:
            recon_error = -1
            print('Wrong ModelName.')
    #print(recon_error_list)
    max_index=recon_res['max_index']
    mem_fea=recon_res['mem_fea'].data
    #print(mem_fea)
    np.save(os.path.join(te_res_path, file_list[file_num] + '.npy'), [tensor.cpu().numpy() for tensor in recon_error_list])
    np.save(os.path.join(te_res_path, file_list[file_num]+'_max_index' + '.npy'), max_index)
    np.save(os.path.join(te_res_path, file_list[file_num] + '_mem_fea' + '.npy'), mem_fea.cpu().numpy())
    np.save(os.path.join(te_res_path, file_list[file_num] + '_fea_list' + '.npy'), [tensor.cpu().numpy() for tensor in fea_list])
utils.eval_video(csv_dataset, te_res_path,file_list[file_num], is_show=False)
utils.eval_max_index(csv_dataset, te_res_path,file_list[file_num])