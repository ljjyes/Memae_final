import numpy as np
import csv
from torch.utils.data import Dataset,DataLoader
import torch
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from joblib import Memory #可以直接用joblib包
from sklearn.datasets import load_svmlight_file
#mem = Memory("./utils/")

#@mem.cache
def dataLoading(path):
    # loading data
    x=[]
    labels=[]
    with (open(path,'r') ) as data_from:
        csv_reader=csv . reader(data_from)
        header=next(csv_reader)
        byte_num = len(header)-2
        print(byte_num)
        x.append(header[0:byte_num])
        labels.append(header[byte_num])
        for i in csv_reader:
            x.append(i[0:byte_num])
            labels.append(i[byte_num])

    for i in range(len(x)):
        for j in range(byte_num):
            x[i][j] = float(x[i][j])
    for i in range(len(labels)):
        labels[i] = float(labels[i])
    x = np.array(x,dtype=np.float32)
    labels = np.array(labels)
    return x,labels

class CsvDataset(Dataset):
    def __init__(self, filepath=r"D:\工程文件，py作业\pythonProject1\buyou_test\dataset\arrhythmia_normalization.csv"):

        print(f"reading {filepath}")  # 打印日志

        feat,label=dataLoading(filepath)

        self.x = torch.from_numpy(feat)
        self.y = torch.from_numpy(label)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

def aucPerformance(mse, labels): #最终评分
    #print(labels[0:12])
    #print(mse[0:12])
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap

def writeResults(name, n_samples_trn,  n_outliers, n_samples_test,test_outliers ,test_inliers, avg_AUC_ROC, avg_AUC_PR, std_AUC_ROC,std_AUC_PR, path):
    #记录结果
    csv_file = open(path, 'a')
    row = name + ","  + n_samples_trn + ','+n_outliers  + ','+n_samples_test+','+test_outliers+','+test_inliers+','+avg_AUC_ROC+','+avg_AUC_PR+','+std_AUC_ROC+','+std_AUC_PR + "\n"
    csv_file.write(row)


if __name__ == '__main__':
    batch_size=512
    csv_dataset = CsvDataset()
    csv_dataloder = DataLoader(csv_dataset, batch_size=batch_size, shuffle=False)

    for idx, (batch_x, batch_y) in enumerate(csv_dataloder):
        print(f'batch_id:{idx},{batch_x.shape},{batch_y.shape}')
        print(batch_x, batch_y)
