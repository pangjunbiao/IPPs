# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 10:43:30 2018

@author: anjing.hu
"""

import numpy as np
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import pickle
from sklearn.neighbors import KDTree
from sklearn.linear_model import LinearRegression


def loadTraceInformation(root):
    info = []
    
    time_lists = os.listdir(root)
    for time_file in time_lists:
        root2 = root + time_file + '/'+ 'AW5992' +'/'
        car_lists = os.listdir(root2)
        for car_file in car_lists:
            file_path = root2 + car_file
            f_data = open(file_path)
            lines = f_data.readlines()
            
            temp = []
            temp.append(int(float(time_file)))
            temp.append(int(float(car_file[-4-4:-4])))
            temp.append(len(lines))
            
            info.append(temp)
            
    return info

def interpolation_GPmodel_building(x_train, y_train, x_test):
    
    x_train = np.atleast_2d(x_train) # bus GPS data
    y_train = np.atleast_2d(y_train).T # cum_dist
    '''kernel 设置'''
    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
    + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
    
    clf = GaussianProcessRegressor(kernel=kernel, alpha=0.0, optimizer='fmin_l_bfgs_b',
                                  n_restarts_optimizer=3, normalize_y=False, random_state=None)
                                  
                                  
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test, return_std=False, return_cov=False)
    
    
    return y_pred,clf
    

def interpolation_linearRegression_KDTree(x_test,data_path):
    
    #data_path = 'D:\\Bus LowRank/likai data/coach data/cluster_12_3-12/the base trace of GP model/AW5992.txt'
    #data_path = 'D:\\wzk bus stop point\\毕业论文-第4章-公交车异常停车点\\3程序\\1程序\\cluster_12_3-12\\the base trace of GP model\\AW5992.txt'
    data = loadData(data_path)
    x_train = data[:,0:2]
    y_train = data[:,2]
        
    y_pred = []
    tree = KDTree(x_train)
    '''kd-Tree 找到近邻'''
    tree_dist,idx_nearest = tree.query(x_test, k=3) 
    n_row = np.size(x_test,axis = 0)
    for idx_test in np.arange(n_row):
        idx_choose = idx_nearest[idx_test,:]
        x_train_temp = x_train[idx_choose,:]
        y_train_temp = y_train[idx_choose]
        '''使用待预测的数据对应的近邻，线性回归得到相应的预测值'''
        lr = LinearRegression()
        lr.fit(x_train_temp, y_train_temp)

        x_test_temp = np.atleast_2d(x_test[idx_test])
        y_pred.append(lr.predict(x_test_temp))
    
    
    return np.array(y_pred)

  

def loadData(data_path):
    data = []
    with open(data_path) as file:
        for line in file:
            tokens = line.strip().split(' ')
            temp = [float(tk) for tk in tokens[1:3]] #lon,Lat
            temp.append(float(tokens[7])) #cum_distance
            data.append(temp)
    data = np.array(data)
   
    return data
    
def loadData_cluster(data_path):
    data = []
    with open(data_path) as file:
        for line in file:
            tokens = line.strip().split('\t')
            temp = [float(tk) for tk in tokens[0:2]] #lon,Lat
            temp.append(float(tokens[3])) #cum_distance
            data.append(temp)
    data = np.array(data)
   
    return data


if __name__ == '__main__':
#    root = './anjing_data/time_gps_txt_new_12_3-12/12_3-12/'
#    info = loadTraceInformation(root)
#    info = np.array(info)
    
    ''' 通过直方图选择路线轨迹 '''
#    plt.figure()
#    n_age, bins_ages, patches_ages = plt.hist(info[:,2],bins = 20,normed=1)
#    title = plt.title('The influence of info')
#    xlabel = plt.xlabel('info')
#    ylabel = plt.ylabel('Proportion')
    
    '''通过直方图找到大致的众数， 最后选择 20171204 AW5992 作为GP的线路轨迹基础'''
    data_path = './the base trace of GP model/AW5992.txt'
    data = loadData(data_path)
    x_train = data[:,0:2]
    y_train = data[:,2]
    n_all = len(y_train)
    
    x_train_norm = scale(x_train, axis=0, with_mean=True, with_std=True, copy=True) 
    x_mean = np.mean(x_train,axis = 0)

    y_max = np.max(y_train)
    y_train_norm = y_train/y_max
    
#    test_data_path = './anjing_data_2/data/affinityPropagation/3.txt'
    test_data_path = './anjing_data_3/data/affinityPropagation/3.txt'

    data = loadData_cluster(test_data_path)
    x_test = data[:,0:2]
    x_test_norm = x_test - x_mean
    
#    idx_test = np.random.randint(200,size = 10)
#    x_test = x_train_norm[idx_test,:]
#    y_test = y_train_norm[idx_test]
#    x_train = np.delete(x_train_norm,idx_test,0)
#    y_train = np.delete(y_train_norm,idx_test)
    
#    y_pred,clf= interpolation_GPmodel_building(x_train_norm, y_train_norm, x_test_norm)
    y_pred_meter = interpolation_linearRegression_KDTree(x_test,data_path)
    
#    y_pred_meter = y_pred * y_max
    y_test_meter = data[:,2]
#    y_test_meter = y_test * y_max
    
    y_test_meter = y_test_meter[:,np.newaxis]
    a = np.abs(y_test_meter - y_pred_meter)
    a = np.hstack((y_pred_meter,y_test_meter,a))
    
    ''' 将建立的模型保存为.pkl 之后直接调用即可'''
#    saveRoot = './GPmodeling/'
#    if not os.path.exists(saveRoot):
#        os.makedirs(saveRoot)
#        
#    savePath = saveRoot + 'GPmodeling.pkl'
#    data = {}
#    data['clf'] = clf
#    data['y_max'] = y_max
#    data['x_mean'] = x_mean
#    f_save = open(savePath,'wb')
#    pickle.dump(data,f_save)
#    f_save.close()
    

        