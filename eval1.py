import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import dataset
import scipy.io as sio
from options.testing_options import TestOptions
import argparse
import utils
import time
from models import  AutoEncoderMem,feature_coder
#绘制图片等
opt_parser = TestOptions()
opt = opt_parser.parse(is_print=True)
file_list = ['arrhythmia_normalization', 'cardio_normalization', 'fraud_normalization',
             'mammography_normalization', 'nslkdd_normalization', 'satellite_normalization',
             'shuttle_normalization', 'spambase_normalization']
file_num=opt.Filenum
#data_root = opt.DataRoot  +'/test'+ '/test_'
data_root = opt.DataRoot  +'/train'+ '/train_'
file_path = data_root + file_list[file_num] + '.csv'
csv_dataset = utils.CsvDataset(filepath=file_path)

model_setting = utils.get_model_setting(opt)
te_res_root = opt.OutRoot
te_res_path = te_res_root + '/' + 'res_' + model_setting

utils.eval_max_index(csv_dataset, te_res_path,file_list[file_num])