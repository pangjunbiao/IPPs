# -*- coding: utf-8 -*-
"""
Created on Wed May 23 11:48:48 2018

@author: PC
"""
import numpy as np
import os
import shutil
import interpolation_GPS as itGPS
import pickle

'''===================读入数据==========================='''
def loadData(path):
    f = open(path)
    data = []
    # R_data 经度 纬度 停留时间 距起点距离 速度距上一点距离 对应红绿灯label 日期 时间 车牌号
    
    # I_data  1、剔除与红绿灯距离远且速度先增后减又在【红绿灯停下】的点（同一辆车在一段范围内有多个停车点）
    #         2、剔除与红绿灯距离近且速度逐渐增大的停车点
    #         3、某些因为GPS点不够而无法计算平均速度的点 modified By anjing
    #         4、剔除速度过大的停车点 # 速度大于150的剔除（第四种情况）
    #         5、剔除速度一直很小的点      

    for line in f.readlines():
        tokens = line.split()
        temp = [float(tt) for tt in tokens]
        data.append(temp)
        
    data = np.array(data)
    
    return data


def GPModle_prediect(gps,model_data):
    
    gpModel = model_data['clf']
    x_mean = model_data['x_mean']
    y_max = model_data['y_max']
    
    ''' rescale '''
    x_norm = gps - x_mean
    y_predict = gpModel.predict(x_norm)
    distance = y_predict * y_max
    
    return distance

    
   
''' ====================构建R矩阵 
计算cumDistance的method1: 使用相对于每条trace的起点距离作为参考，快速验证 结果并不准确！！！
计算cumDistance的method2: 选取一条trace作为参考路径，使用该trace上的GPS点使用GP建模。cumDistance,通过GP模型预测。
                          结果不准确。可能由于GP的kernel选择问题。
计算cumDistance的method3: 选取一条trace作为参考路径。余下所有GPS点，每个GPS点使用KDTree找到参考路径中的k-nearest点，
                          使用这些k-nearest点进行linearRegression建模，预测每个GPS的cumDistance。较为准确。
====================== '''
def fillMatrixRandI(ex, cluster_idx, dis_total_max_min, R_data,I_data,range_para,opt_type, data_path):
    '''形成R,I,label'''
    num_stopPoint = np.size(R_data,axis = 0)
    '''====method1==== '''
    #    dis_total = R_data[:,3]
    
    '''====method2===='''
    #    model_save_path = './GPmodeling/GPmodeling.pkl'
    #    model_file = open(model_save_path,'rb')
    #    model_data = pickle.load(model_file)
    #
    #    gps_total = R_data[:,0:2]
    #    dis_total = GPModle_prediect(gps_total,model_data)
    '''====method3===='''

    gps_total = R_data[:,0:2]
    dis_total = itGPS.interpolation_linearRegression_KDTree(gps_total,  data_path)
    if ex == False:
        #去除聚类17中的异常值
        if cluster_idx == 17:
            ind = list(np.where(dis_total > 10000))
            dis_total_1 = dis_total[dis_total < 10000]
            dis_total = dis_total_1
            R_data = np.delete(R_data, ind, axis=0)
            num_stopPoint = np.size(R_data, axis=0)

    true_GPS = np.array([[116.302661, 39.880903],
                [116.310097, 39.897319],
                [116.310297, 39.925278],
                [116.310304, 39.938717],
                [116.309811, 39.943409],
                [116.309557, 39.953169],        #1 该点在聚类之外 改   2,W矩阵该点在聚类之外
                [116.313866, 39.964449],
                [116.380633, 39.981016],       #2 该点在聚类之外 改
                [116.366219, 40.007226],
                [116.362843, 40.011537]])
    # 真实的停车点到起点的距离
    dis_true = itGPS.interpolation_linearRegression_KDTree(true_GPS, data_path)
    # 第cluster_idx聚类的区间长度
    dis_total_range = dis_total_max_min[cluster_idx-1,1] - dis_total_max_min[cluster_idx-1,0]
    # 分成range_para长度的小块的个数
    num_seg = int(np.floor(dis_total_range/range_para))
    # 初始化label为num_seg+1,因为长度除以range_para后可能还有余数
    true_label = [0]*(num_seg + 1)
    # 记录在当前聚类里面的真实停车点在小块中的位置
    index_true = []
    for true_i in range(len(dis_true)):
        # 循环判断所有停车点，看哪些在当前聚类区间里面
        if dis_true[true_i] >= dis_total_max_min[cluster_idx-1,0] and dis_true[true_i] <= dis_total_max_min[cluster_idx-1,1]:
            # 真实停车点到聚类起点的距离
            dis_true_ori = dis_true[true_i] - dis_total_max_min[cluster_idx-1,0]
            # 停车点在哪个小块
            index_true.append(int(np.floor(dis_true_ori/range_para)))

    # 循环所有小块，停车点位置
    for seg_i in range(num_seg+1):
        for ind_j in range(len(index_true)):
            # 得到当前聚类的label
            if seg_i == index_true[ind_j]:
                true_label[seg_i] = 1

    #print('true_label:', true_label)

    '''==============='''
    
    dis_max = max(dis_total)
    dis_min = dis_total_max_min[cluster_idx - 1, 0] #min(dis_total)
    dis_range = dis_total_max_min[cluster_idx - 1, 1] - dis_total_max_min[cluster_idx - 1, 0]# dis_max - dis_min
    dis_total_norm = np.copy(dis_total)
    dis_total_norm = dis_total_norm - dis_min    
    if opt_type == 'meter':
        range_meter = range_para
    elif opt_type == 'times':
        range_meter = dis_range*1.0/range_para    
        
    '''#每个停车点都划分到不同的路段中，dis_range_idx表示每个停车点对应的路段index'''
    dis_range_idx = np.floor(dis_total_norm/range_meter)

    num_segmentation = 1 + np.ceil(dis_range/range_meter) #正无穷大取整
    num_segmentation = int(num_segmentation)
    
    '''flag_meter: 将整段路线分段，每个节点距离原点的距离'''
    flag_meter = np.ones((1,num_segmentation))*range_meter
    flag_meter = np.cumsum(flag_meter)
    flag_meter = flag_meter - range_meter
    flag_meter[num_segmentation - 1] = dis_range
       
    
    temp_R = np.zeros(num_segmentation -1)
    temp_I = np.zeros(num_segmentation -1)

    flag_date = R_data[0,7]
    flag_plate = R_data[0,9]
    R_matrix = []
    I_matrix = []
    info_matirx = []
    flag_idx = []
    for idx in np.arange(num_stopPoint):       
        
        ''' 一辆车一天对应矩阵的一行 '''
        if R_data[idx,7] != flag_date or R_data[idx,9] != flag_plate:
            R_matrix.append(temp_R)
            I_matrix.append(temp_I)
            info_matirx.append(np.array([flag_plate,flag_date]))
            
            flag_date = R_data[idx,7]
            flag_plate = R_data[idx,9]
            
            temp_R = np.zeros(num_segmentation-1)
            temp_I = np.zeros(num_segmentation-1)

            
        idx_temp = int(dis_range_idx[idx])
        flag_idx.append(idx_temp)
        
        '''===============构造I_matrix=================='''
        if I_data[idx,0] > 0 or I_data[idx,1] > 0:
            if idx_temp == num_segmentation-1:
                temp_I[idx_temp-1] = 1.0
            else:                    
                temp_I[idx_temp] = 1.0
        
        '''===============构造R_matrix================='''
        ''' 首尾两停车点的时间处理 '''
        if idx_temp == 0 :
            temp_R[idx_temp] += R_data[idx,2]
        elif idx_temp == num_segmentation-1:
            temp_R[idx_temp-1] += R_data[idx,2]
        elif idx_temp == num_segmentation-2:
            temp_R[idx_temp-1] += R_data[idx,2]            
        else:
            '''将停车时间voting到两个时间段中'''
            
            weight_font = (flag_meter[idx_temp + 1] - dis_total_norm[idx])*1.0/(flag_meter[idx_temp + 1] - flag_meter[idx_temp])
            weight_back = (dis_total_norm[idx] - flag_meter[idx_temp] )*1.0/(flag_meter[idx_temp + 1] - flag_meter[idx_temp])


            if flag_meter[idx_temp + 1] - range_para/2 - dis_total_norm[idx] > 0:
                weight_font = (flag_meter[idx_temp + 1] - range_para/2 - dis_total_norm[idx])*1.0/(flag_meter[idx_temp + 1] - flag_meter[idx_temp])
                weight_back = (dis_total_norm[idx] - flag_meter[idx_temp ] +50 )*1.0/(flag_meter[idx_temp + 1] - flag_meter[idx_temp])
                
                ''' 将权重归一化 '''            
                temp_R[idx_temp] += weight_back*R_data[idx,2]
                temp_R[idx_temp - 1] +=  weight_font*R_data[idx,2] 
            else:
                weight_font = (- flag_meter[idx_temp ] - range_para/2 + dis_total_norm[idx])*1.0/(flag_meter[idx_temp + 1] - flag_meter[idx_temp])
                weight_back = (flag_meter[idx_temp + 1] - dis_total_norm[idx] + range_para/2 )*1.0/(flag_meter[idx_temp + 1] - flag_meter[idx_temp])
                
                ''' 将权重归一化 '''            
                temp_R[idx_temp + 1] += weight_back*R_data[idx,2]
                temp_R[idx_temp] +=  weight_font*R_data[idx,2] 
                
      
    R_matrix.append(temp_R)
    R_matrix = np.array(R_matrix)     
    info_matirx.append(np.array([flag_plate,flag_date]))    
    
    I_matrix.append(temp_I)
    I_matrix = np.array(I_matrix)
    info_matirx = np.array(info_matirx)
    flag_idx = np.array(flag_idx)
    flag_idx = flag_idx[:,np.newaxis]
    
    return R_matrix,I_matrix,info_matirx,flag_idx, true_label
    

        
''' ===============将矩阵保存至txt =================='''        
def writeIntoTxt(matrix,savePath):
    f = open(savePath,'w')
    n_row = np.size(matrix,axis = 0)
    for i in np.arange(int(n_row)):
        for j in matrix[i,:]:
            f.write(str(j))
            f.write('\t')
        f.write('\n')
        
    f.close()



def BuildingMatrixRandI(ex, cluster_idx, np_len, R_data,I_data,range_para,opt_type,savePath_R,savePath_I,savePath_info,savePath_flag_GPS):
    data_path = './the base trace of GP model/AW5992.txt'
    if opt_type == 'meter':
        R_matrix,I_matrix,info_matirx,flag_idx, true_label = fillMatrixRandI(ex, cluster_idx, np_len, R_data,I_data,range_para,opt_type, data_path)
        
    elif opt_type == 'times':
        R_matrix,I_matrix,info_matirx,flag_idx = fillMatrixRandI(R_data,I_data,range_para,opt_type, data_path)

    writeIntoTxt(R_matrix,savePath_R)
    writeIntoTxt(I_matrix,savePath_I)
    writeIntoTxt(info_matirx,savePath_info)
    writeIntoTxt(flag_idx,savePath_flag_GPS)
    
    return R_matrix, true_label
    
    
if __name__ == '__main__'  :
    ex = False                 #if V矩阵：ex=Ture  else  W矩阵： ex=False
    range_para_list = [100,200,300,400]
    for range_para in range_para_list:
        #print('--------------------------------')
        opt_type = 'meter'
        Root = './result matrix/'               #V矩阵：'./result matrix_ex/'   W矩阵'./result matrix/'
        saveRoot = Root + 'matrix_' + opt_type+ str(range_para)+'/'
        saveRoot_R = saveRoot + 'R/'
        saveRoot_I = saveRoot + 'I/'
        saveRoot_info = saveRoot + 'info/'
        saveRoot_flag_GPS = saveRoot+'flag_GPS/'
        if not os.path.exists(saveRoot_R):
            os.makedirs(saveRoot_R)
        if not os.path.exists(saveRoot_I):
            os.makedirs(saveRoot_I)
        if not os.path.exists(saveRoot_info):
            os.makedirs(saveRoot_info)
        if not os.path.exists(saveRoot_flag_GPS):
            os.makedirs(saveRoot_flag_GPS)
    
        
                    
        dataRoot = './anjing_data_3/data/affinityPropagation/'         #V矩阵'./anjing_data_3ex/data/affinityPropagation/'   W矩阵#V矩阵'./anjing_data_3/data/affinityPropagation/'
        cluster_idxs = range(1,24)#24   #V矩阵23

        all_true_label = []

        lenl = []
        data_path = './the base trace of GP model/AW5992.txt'

        true_GPS = np.array([[116.302661, 39.880903],
                             [116.310097, 39.897319],
                             [116.310297, 39.925278],
                             [116.310304, 39.938717],
                             [116.309811, 39.943409],
                             [116.309557, 39.953169],
                             [116.313866, 39.964449],
                             [116.380633, 39.981016],
                             [116.366219, 40.007226],
                             [116.362843, 40.011537]])
        # 10个实际停车点与原点的距离
        dis_true = itGPS.interpolation_linearRegression_KDTree(true_GPS, data_path)
        # print(dis_true)

        # 得到每个聚类的范围，每个聚类起始和终止点距离原点的距离,[[聚类1起始-原点,聚类2终止-原点],[],[]...]
        for cluster_idx in cluster_idxs:
            cluster_txt = str(cluster_idx) + '.txt'
            dataPath1 = dataRoot + cluster_txt
            R_data = loadData(dataPath1)
            gps_total = R_data[:, 0:2]
            # 用gps坐标求距离
            dis_total = itGPS.interpolation_linearRegression_KDTree(gps_total, data_path)
            # 求W,E时候去除聚类17的异常值
            if ex == False:
                if cluster_idx == 17:
                    ind = list(np.where(dis_total > 10000))
                    dis_total_1 = dis_total[dis_total < 10000]
                    dis_total = dis_total_1
            # 记录聚类中最大最小距离
            leni = []
            leni.append(min(dis_total).item())
            leni.append(max(dis_total).item())
            lenl.append(leni)
        np_len = np.array(lenl)
        # print(np_len)

        # 排序数组（根据起始点）寻找完全重叠的聚类, 手动看哪些聚类被完全包含，写为下面数组
        sorted_indices = np.argsort(np_len[:, 0])
        sorted_data = np_len[sorted_indices]


        if ex == False:
            # 被包含的聚类
            out_cluster_idx = [6, 10, 16, 18, 19]
        else:
            out_cluster_idx = [11, 6, 10, 13]

        # 去除之后看是否不含被完全包含的聚类
        array_without_selected_rows = np.delete(np_len, out_cluster_idx, axis=0)
        sorted_indices1 = np.argsort(array_without_selected_rows[:, 0])
        sorted_data1 = array_without_selected_rows[sorted_indices1]

        # 两个聚类存在小的交叠时用前一个终点替换后一个起点
        need_tem = []
        tem = []
        for ii in range(len(sorted_data1)-1):
            if sorted_data1[ii+1][0] < sorted_data1[ii][1]:
                need_tem.append(sorted_data1[ii+1][0])
                tem.append(sorted_data1[ii][1])
        for ii in range(len(need_tem)):
            ind_need_tem = np.where(np_len==need_tem[ii])
            np_len[ind_need_tem[0][0]][ind_need_tem[1][0]]=tem[ii]

        array_without_selected_rows111 = np.delete(np_len, out_cluster_idx, axis=0)
        sorted_indices111 = np.argsort(array_without_selected_rows111[:, 0])
        sorted_data111 = array_without_selected_rows111[sorted_indices111]

        if ex == False:
            #7, 11, 17, 19, 20为完全包含关系，可以去除，0,2,3,4,5为五环以外的聚类
            out_cluster = [7, 11, 17, 19, 20, 0, 2, 3, 4, 5]
        else:
            out_cluster = [7, 11, 12, 14, 1, 6, 8, 14, 16, 19]

        # 对每个聚类求R,I,label
        for cluster_idx in cluster_idxs:
            #去除完全重叠的聚类
            if cluster_idx in out_cluster: continue
            #print('cluster_idx:', cluster_idx)

            cluster_txt = str(cluster_idx) + '.txt'
            dataPath1 = dataRoot + cluster_txt
            R_data = loadData(dataPath1)
            
            dataPath2 = './I_idx/' + cluster_txt     #V矩阵'./I_idx_ex/'   W矩阵'./I_idx/'
            I_data = loadData(dataPath2)
            
            savePath_R = saveRoot_R+'R_matrix_of_cluster' +cluster_txt
            savePath_I = saveRoot_I+'I_matrix_of_cluster' +cluster_txt
            savePath_info = saveRoot_info +'info_matrix_of_cluster' +cluster_txt
            savePath_flag_GPS = saveRoot_flag_GPS + 'GPS_flag_of_cluster' +cluster_txt
            R_matrix, true_label = BuildingMatrixRandI(ex, cluster_idx, np_len, R_data,I_data,range_para,opt_type,savePath_R,savePath_I,savePath_info,savePath_flag_GPS)
            for s in range(len(true_label)):
                all_true_label.append(true_label[s])

            # print(len(true_label))
        print('all_true_label_'+str(range_para), all_true_label)
        print(sum(all_true_label))


#    old_data_path = './result matrix'   
#    new_data_path = '.../trys_mat/data_matrix'
#    os.popen("cp /old_data_path/* /new_data_path").read()






    
