# -*- coding: utf-8 -*-
import numpy as np
import sys
import FromTxtLoadData

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from itertools import cycle
import time
import os

from scipy.stats import norm
from sklearn.neighbors import KernelDensity

def oCluster(file_path):
    data = FromTxtLoadData.readDataFromTxt(file_path)
    o_day = np.array(data[:, 0], dtype=float)
    
    o_time = np.array(data[:, 1], dtype=float)
    o_arv = np.array(data[:, 2], dtype=float)
    o_lon = np.array(data[:, 3],dtype = float)
    o_lat = np.array(data[:, 4], dtype=float)
    o_v = np.array(data[:, 5], dtype=float)
    o_dis_up = np.array(data[:, 6], dtype=float)
    o_dis_TrafficSignal = np.array(data[:, 7], dtype=float)
    o_TrafficSignal_label = np.array(data[:, 8], dtype=float)
    o_dis_total = np.array(data[:, 9], dtype=float)
    o_plate = np.array(data[:, 10],dtype = float)
    o_stay_time = np.array(data[:,11],dtype=float)
    
    
    '''# 经度 纬度 停留时间 距起点距离 速度 距上一点距离 对应红绿灯label 日期 时间 车牌号 到达聚类时间'''
    o_data = np.zeros([(o_lon.shape)[0], 11])
    o_data[:, 0] = o_lon
    o_data[:, 1] = o_lat
    o_data[:, 2] = o_stay_time
    o_data[:, 3] = o_dis_total
    o_data[:, 4] = o_v
    o_data[:, 5] = o_dis_up
    o_data[:, 6] = o_TrafficSignal_label
    o_data[:, 7] = o_day
    o_data[:, 8] = o_time
    o_data[:, 9] = o_plate
    o_data[:, 10] = o_arv
    return o_data

def oCluster2(file_path):
    data = FromTxtLoadData.readDataFromTxt_value(file_path)
    o_lon = np.array(data[:, 0], dtype=float)
    o_lat = np.array(data[:, 1],dtype = float)
    o_time = np.array(data[:, 3],dtype = float)
    o_plate = np.array(data[:, 5],dtype = float)
    o_data = np.zeros([(o_lon.shape)[0], 4])
    o_data[:, 0] = o_lon
    o_data[:, 1] = o_lat
    o_data[:, 2] = o_time
    o_data[:, 3] = o_plate
    return o_data

def KDE(center_time, txt_name, new_name, color):
    plt.figure(1) #新建一个为figure1的画图窗口
    X = center_time / 3600
    X_plot = np.linspace(5, 15, 1000)[:, np.newaxis]
    # width = [0.1, 0.2, 0.3, 0.4, 0.5]
    width = [0.3]
    for bandwidth in width:
        for kernel in ['gaussian']:
            kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(X)
            log_dens = kde.score_samples(X_plot)
            plt.plot(X_plot[:, 0], np.exp(log_dens), '-',
                    label="kernel = '{0}'".format(kernel))

        plt.legend(loc='upper left')
        plt.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), '+k')
        plt.xlim(5,14)
        plt.ylim(-0.02, 0.8)
        plt.title('bandwidth : {}'.format(str(bandwidth)))
        plt.xlabel('hour')
        # plt.show()
        plt.savefig('./kde_{}_{}_{}.jpg'.format(str(txt_name), in_or_out, str(bandwidth)))
        plt.clf()

def dbScan(o_data, save_dir, in_or_out):
    db = DBSCAN(eps=0.005, min_samples=10).fit(o_data[:,0:2])
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    num = 1
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        xy = o_data[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=5)

        xy = o_data[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=5)
        filename = save_dir + '/' + str(num) + '.txt'
        f = open(filename, "w")
        for t in range(len(xy)):
            f.write('\t'.join([str(tt) for tt in xy[t, :]]))
            f.write("\n")

        f.close()
        num += 1

    plt.title('dbScan: Estimated number of clusters: %d' % n_clusters_)
    # plt.show()
    plt.savefig('./dbScan_0.005_10_{}.jpg'.format(in_or_out))


def affinityPropagation(o_data, save_dir, in_or_out, f_center):
    af = AffinityPropagation(damping=.9, max_iter=200).fit(o_data[:,0:2])
    core_samples_mask = np.zeros_like(af.labels_, dtype=bool)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters_ = len(cluster_centers_indices)

    plt.close('all')
    plt.figure(1)
    plt.clf()

    center_time = []
    center_lon = []
    center_lat = []
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    num = 1
    # filename = save_dir + '/' + str(num) + '.txt'
    # f = open(filename, "w")
    for k, col in zip(range(n_clusters_), colors):
        class_members = labels == k
        cluster_center = o_data[cluster_centers_indices[k]]

        # 聚类中心
        print(str(k + 1) + '   ' + str(cluster_center[1]) + ' , ' + str(cluster_center[0]) + ' , ' + str(cluster_center[3]) + ' , ' + str(cluster_center[4]))

        plt.plot(o_data[class_members, 0], o_data[class_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
        for x in o_data[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
        # plt.show()
        class_member_mask = (labels == k)
        test_idx = class_member_mask & core_samples_mask
        xy = o_data[class_member_mask & core_samples_mask]
        xy = o_data[class_member_mask & ~core_samples_mask]
        print('max lon:{},max lat:{},min lon:{},min lat:{}'.format(str(max(xy[:,0])), str(max(xy[:,1])), str(min(xy[:,0])), str(min(xy[:,1]))))
        
        '''聚类写入文件'''
        filename = save_dir + '/' + str(num) + '.txt'
        f = open(filename, "w")
        for t in range(len(xy)):
            f.write('\t'.join([str(tt) for tt in xy[t, :]]))
            f.write("\n")

        f.close()
        num += 1
        # 经度 纬度 停留时间 日期 时间
        f_center.write(str(cluster_center[0]))
        f_center.write(',')
        f_center.write(str(cluster_center[1]))
        f_center.write(',')
        f_center.write(str(cluster_center[2]))
        f_center.write(',')
        f_center.write(str(cluster_center[3]))
        f_center.write(',')
        f_center.write(str(cluster_center[4]))
        f_center.write('\n')
        center_time.append(np.array([cluster_center[4]]))
        center_lon.append(np.array([cluster_center[0]]))
        center_lat.append(np.array([cluster_center[1]]))

    plt.title('affinityPropagation: Estimated number of clusters: %d' % n_clusters_)
    # plt.show()
    plt.savefig('./affinityPropagation_.9_{}.jpg'.format(in_or_out))
    return np.array(center_time), np.array(center_lon), np.array(center_lat)

if __name__ == '__main__':
    anjingInput_root = '../pretreatment/'
        
    anjingOutput_root = './anjing_data_3ex/'
    if not os.path.exists(anjingOutput_root):
        os.mkdir(anjingOutput_root)
        
    txt_dir = anjingOutput_root + 'data/'
    if not os.path.exists(txt_dir):
        os.mkdir(txt_dir)
    txt_dir = txt_dir + 'affinityPropagation'
    if not os.path.exists(txt_dir):
        os.mkdir(txt_dir)
    
    in_or_out = 'out'
    o_data = oCluster(anjingInput_root + 'ex_new_3-12.txt')
    cluster_center_path = anjingOutput_root +'cluster_cernter.txt'
    f_center = open(cluster_center_path, 'w')

    '''draw'''
    # dbScan(o_data, './data/dbscan',in_or_out)
    center_time, center_lon, center_lat = affinityPropagation(o_data, txt_dir,in_or_out, f_center)
    f_center.close()
    colors = ['red', 'blue', 'darkcyan', 'yellowgreen', 'indianred', 'purple', 'gray', 'mediumpurple',
              'darkgreen', 'darkred', 'goldenrod', 'green', 'yellow', 'pink', 'cyan', 'lightgreen', 'gold', 'brown']

    '''每一个聚类点周围停车点的时间分布'''
    # txt_name_list = [1,2,3,4,5,6,7,8,9,10,11]
    # new_list = [1,9,6,3,4,11,8,5,2,7,10]
    # for txt_name_idx in range(len(txt_name_list)):
    #     txt_name = txt_name_list[txt_name_idx]
    #     luzhuangqiao_path = './data/affinityPropagation/{}.txt'.format(str(txt_name))
    #     f_lz = open(luzhuangqiao_path)
    #     lines = f_lz.readlines()
    #     lz = []
    #     for line in lines:
    #         lz.append(np.array([float(line.split('\t')[4])]))
    #     lz_np = np.array(lz)
    #
    #     KDE(lz_np, txt_name, new_list[txt_name_idx], colors[txt_name_idx])

    
    txt_list = os.listdir(txt_dir)
    for txt_name in txt_list:
        path = txt_dir + '/' + txt_name
        data = oCluster2(path)
        data_sort = data[np.lexsort(data.T)]
        day = 0
        plate = 0
        num = 0
        for line_ in range(len(data_sort)):
            txt_day = int(data_sort[line_,2])
            txt_plate =  int(data_sort[line_,3])
            if txt_day != day or txt_plate != plate:
                # print (str(txt_day) + ' ' + str(txt_plate))
                num += 1
                day = txt_day
                plate = txt_plate
        print (txt_name + ': ' + str(num))

    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # X = center_lon[:,0]
    # Y = center_lat[:,0]
    # Z = center_time[:,0]/3600
    # ax.scatter(X, Y, Z)
    # plt.show()