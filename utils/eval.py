import os
import scipy.io as sio
import numpy as np
import sklearn.metrics as skmetr
import utils
import matplotlib.pyplot as plt
import pyecharts.options as opts
from options.testing_options import TestOptions
from pyecharts.charts import Scatter3D
#rauc[i], ap[i] = aucPerformance(scores, test_label) 直接给出两个指标，只要给出sores和label就行
# 只需要在顶部声明 CurrentConfig.ONLINE_HOST 即可
from pyecharts.globals import CurrentConfig, OnlineHostType
# OnlineHostType.NOTEBOOK_HOST 默认值为 http://localhost:8888/nbextensions/assets/
CurrentConfig.ONLINE_HOST = "http://127.0.0.1:8000/assets/"
from pyecharts.charts import Scatter3D, Page
from pyecharts import options as opts
from pyecharts.globals import ThemeType
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot

def eval_video(data_set, res_path,file_name, is_show=False):

    res_file_name = file_name + '.npy'
    res_file_path = os.path.join(res_path, res_file_name)
    #print(res_file_path)
    res_prob = np.load(res_file_path,allow_pickle=True).astype(np.float32)

    res_prob_list = []
    res_prob_list_org = []

    res_prob_list_org = res_prob_list_org + list(res_prob)
    res_prob_norm = np.subtract(res_prob ,res_prob.min())
    res_prob_norm = res_prob_norm / res_prob_norm.max()

    res_prob_list = res_prob_list + list(res_prob_norm)

    labels=data_set.y.numpy()
    print(len(labels))
    print(res_prob_list[0:12])

    rauc, ap = utils.aucPerformance(res_prob_list, labels)

def plot_scatter(d1,d2,d3, save_dir): #version 1.5-1.9 
    
    
    
    piece=[
      {'value': 0,'label': 'class A','color':'#e57c27'}, 
      {'value': 1, 'label': 'class B','color':'#72a93f'},
      {'value': 2, 'label': 'class C','color':'#368dc4'}
    ]
 
    sc1 = Scatter3D() 
    sc1.add("", d1.tolist()) 
    sc1.add("", d2.tolist()) 
    sc1.add("", d3.tolist()) 
    sc1.set_global_opts(title_opts=opts.TitleOpts(title="3D scatter"),
        visualmap_opts=opts.VisualMapOpts(dimension=3,is_piecewise=True, pieces=piece))
    sc1.render('scatter3d.html')
    
def eval_max_index(data_set,res_path,file_name):
    res_file_name = file_name+'_max_index' + '.npy'
    res_file_path = os.path.join(res_path, res_file_name)
    max_index = np.load(res_file_path, allow_pickle=True).astype(np.float32)
    max_index=list(max_index)

    res_file_name1 = file_name + '_fea_list' + '.npy'
    res_file_path1 = os.path.join(res_path, res_file_name1)
    fea_list = np.load(res_file_path1, allow_pickle=True)
    fea_list = fea_list

    res_file_name2 = file_name + '_mem_fea' + '.npy'
    res_file_path2 = os.path.join(res_path, res_file_name2)
    mem_fea = np.load(res_file_path2, allow_pickle=True)
    #mem_fea = list(mem_fea)
    #print(fea_list)
    
    labels = data_set.y.numpy()
    print(mem_fea)
    # 配置 config
    '''
    (
        Scatter3D()
        .add(
            series_name="Mem_fea",
            data=mem_fea.tolist(),
            
        )
        .add(
            series_name="fea_list",
            data=fea_list[1:300],
        )
        .set_global_opts(
                title_opts=opts.TitleOpts(title="3D scatter"),
                visualmap_opts=opts.VisualMapOpts(dimension=3)
            )
        .render("scatter3d.html")
    )
    '''
    fig,ax = plt.subplots()
    n,bins_num,pat = ax.hist(max_index,bins=10,alpha=0.75,color='#7DB6F7')
    ax.plot(bins_num[:10]+0.05,n,marker = 'o',color="#F4C2E3",linestyle="--")  
    plt.xlabel('mem_index',size=12)
    plt.ylabel('counts',size=12)
    plt.title('a_b_c_2')
    plt.xticks([4+i*0.1 for i in range(-5,5)],[f'{i}'  for i in range(10)])
    plt.savefig('max_hist.png')
    print(len(max_index))
    print(len(fea_list))
    print(len(mem_fea))


    