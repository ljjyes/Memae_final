import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def data_c(file_path):
    file=pd.read_csv(file_path,header=None)
    print(file.shape)
    #print(file.shape[1]-2)
    groups = file.groupby(file.shape[1]-2)
    subset1 = groups.get_group(0) #正常样本
    subset2 = groups.get_group(1) #异常样本
    print(len(subset2)/file.shape[0])
    return subset1,subset2
def data_prepare(dataseed):
    #由于其中NSL-KDD过于均衡，不太符合异常样本少量的要求，所以用下面的分割方法
    #正常样本2/1分割，2份用于Train，1份用于Test，其中Test需要从异常样本中抽取使异常率保持30%左右
    #OKKK就是这样捏
    file_list = ['arrhythmia_normalization.csv', 'cardio_normalization.csv', 'fraud_normalization.csv',
                 'mammography_normalization.csv', 'nslkdd_normalization.csv', 'satellite_normalization.csv',
                 'shuttle_normalization.csv', 'spambase_normalization.csv']
    file_root = 'dataset/'
    train_root='dataset/train/'
    test_root='dataset/test/'
    for i in range(len(file_list)):
        file_path=file_root+file_list[i]
        nomset,anomset=data_c(file_path)
        #print(nomset.shape)
        train_data,test_data_nom=train_test_split(nomset,test_size=0.33,random_state=dataseed)
        #正常的样本占0.7，在乘0.43后，最后异常的比例为0.3
        test_anom_num=int(test_data_nom.shape[0]*0.43)
        #在异常的样本不能达到0.3时，直接加入所有的异常样本
        #在异常样本超了的时候，维持30%的比例（主要针对NSL-KDD这个数据集）
        if anomset.shape[0] <= test_anom_num:
            test_data=pd.concat([test_data_nom, anomset], axis=0)
        else:
            rate=test_anom_num/anomset.shape[0]
            n1,test_data_anom=train_test_split(anomset,test_size=rate,random_state=dataseed)
            test_data=pd.concat([test_data_nom, test_data_anom], axis=0)
        np.savetxt(f'{train_root}train_{file_list[i]}', np.array(train_data), fmt='%f', delimiter=',')
        np.savetxt(f'{test_root}test_{file_list[i]}', np.array(test_data), fmt='%f', delimiter=',')


if __name__ == '__main__':
    data_prepare(42)