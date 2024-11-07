# -*- coding: utf-8 -*-
"""
Created on Sat Sep 08 17:00:36 2018

@author: Administrator
"""
import numpy as np
import scipy.io as scio

#import roc_curve_k

from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn import metrics
from inspect import signature
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

from sklearn.tree import DecisionTreeClassifier

    

def MaxMinNormalization(x,Max,Min):
    x = (x - Min) / (Max - Min)
    return x



def PR_thresh(y_true, y_score_):
    
#    thresholds = np.random.rand(10000) *0.7
    thresholds = np.linspace(0,1,5)

    pos_label = 1.

    # make y_true a boolean vector
#    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    y_score_ = y_score_.reshape(-1,1)
    desc_score_indices = np.lexsort(y_score_[:,::-1].T)[::-1]
    y_score_ = y_score_[desc_score_indices]
    y_true = y_true[desc_score_indices]
    

    precision_list = []
    recall_list = []
    
    for j in thresholds:
        
        y_score = np.zeros(len(y_score_))
        y_score[np.where(y_score_ > j)[0]] = 1
#        y_score[np.where(a <= j)[0]] = 0

        TP = 0
        FN = 0
        TN = 0
        FP = 0
        for i in np.arange(len(y_score)):
            if y_true[i] == 1:
                if y_score[i] == y_true[i]:
                    TP = TP + 1
                if y_score[i] != y_true[i]:
                    FN = FN + 1
            if y_true[i] == 0:
                if y_score[i] == y_true[i]:
                    TN = TN + 1
                if y_score[i] != y_true[i]:
                    FP = FP + 1
#        print ('TP = ' + str(TP))
#        print ('FN = ' + str(FN))
#        print ('TN = ' + str(TN))
#        print ('FP = ' + str(FP))
        
        if TP+FP == 0:
            precision = 0
        else:
            precision = float(TP)/float((TP+FP))
            
        if TP+FN == 0:
            TPR = 0
        else:
            recall = float(TP)/float((TP+FN))
            TPR = float(TP)/float((TP+FN))
            
        if FP+TN == 0:
            FPR = 0
        else:  
            FPR = float(FP)/float((FP+TN))
            
        if TP+TN == 0:
            TNR = 0
        else:  
            TNR = float(TN)/float((TP+TN))
            
        if TP+FN == 0:
            FNR = 0
        else: 
            FNR = float(FN)/float((TP+FN))
    
#        precision = tp / (tp + fp)
#        precision[np.isnan(precision)] = 0
#        recall = tp / tp[-1]
        precision_list.append(precision)
        recall_list.append(recall)
#    recall_list.sort(reverse = True)
#    precision_list.sort()

#    precision_list = np.unique(np.array(precision_list))
#    precision_list = np.array(precision_list)

    
    return precision_list,recall_list, thresholds


#label可以通过getBoundOfClusters.py得到，W,E,R的标签

all_true_label_100=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
all_true_label_200=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
all_true_label_300=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
all_true_label_400=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]




#选用什么插值距离的标签
np_label = np.array(all_true_label_200)

# V的标签
v_true_label_100=[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
v_true_label_200=[0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
v_true_label_300=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
v_true_label_400=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0]




np_label_v = np.array(v_true_label_200)



E_time_path = "./E_time.mat"
W_time_path = "./W_time.mat"
R_time_path = "./R_time.mat"
V_time_path = "./V_time.mat"



E_time = scio.loadmat(E_time_path)
R_time = scio.loadmat(R_time_path)
W_time = scio.loadmat(W_time_path)
V_time = scio.loadmat(V_time_path)

test_time = "./test_time.mat"
test_time = scio.loadmat(test_time)
test_time = test_time.get('test_time')


W_all_time = W_time.get('E_num_all_cluster')
W_col_time = W_time.get('E_num_max_cluster')
W_max_time = W_time.get('E_num_col_cluster')

E_all_time = E_time.get('E_num_all_cluster')
E_col_time = E_time.get('E_num_max_cluster')
E_max_time = E_time.get('E_num_col_cluster')

R_all_time = R_time.get('E_num_all_cluster')
R_col_time = R_time.get('E_num_max_cluster')
R_max_time = R_time.get('E_num_col_cluster')

V_all_time = V_time.get('E_num_all_cluster')
V_col_time = V_time.get('E_num_max_cluster')
V_max_time = V_time.get('E_num_col_cluster')


W_all_time = MaxMinNormalization(W_all_time, np.max(W_all_time),np.min(W_all_time))
W_col_time = MaxMinNormalization(W_col_time, np.max(W_col_time),np.min(W_col_time))
W_max_time = MaxMinNormalization(W_max_time, np.max(W_max_time),np.min(W_max_time))

E_all_time = MaxMinNormalization(E_all_time, np.max(E_all_time),np.min(E_all_time))
E_col_time = MaxMinNormalization(E_col_time, np.max(E_col_time),np.min(E_col_time))
E_max_time = MaxMinNormalization(E_max_time, np.max(E_max_time),np.min(E_max_time))

R_all_time = MaxMinNormalization(R_all_time, np.max(R_all_time),np.min(R_all_time))
R_col_time = MaxMinNormalization(R_col_time, np.max(R_col_time),np.min(R_col_time))
R_max_time = MaxMinNormalization(R_max_time, np.max(R_max_time),np.min(R_max_time))

V_all_time = MaxMinNormalization(V_all_time, np.max(V_all_time),np.min(V_all_time))
V_col_time = MaxMinNormalization(V_col_time, np.max(V_col_time),np.min(V_col_time))
V_max_time = MaxMinNormalization(V_max_time, np.max(V_max_time),np.min(V_max_time))





all_time_sort = []

all_time_sort.append(W_all_time)
all_time_sort.append(W_col_time)
all_time_sort.append(W_max_time)

all_time_sort.append(E_all_time)
all_time_sort.append(E_col_time)
all_time_sort.append(E_max_time)

all_time_sort.append(R_all_time)
all_time_sort.append(R_col_time)
all_time_sort.append(R_max_time)

all_time_sort.append(V_all_time)
all_time_sort.append(V_col_time)
all_time_sort.append(V_max_time)

#clf = DecisionTreeClassifier()  #决策树分类器
#clf.fit(X_train, y_true)  #拟合数据
#y_pred = clf.predict(X_train)  #得出预测结果,从测试数据集

E_all_time = np.delete(test_time, [3], axis = 0)
E_all_time = MaxMinNormalization(E_all_time, np.max(E_all_time),np.min(E_all_time))

k = 10  #取前10个可能的停车点
for i in range(len(all_time_sort)):
    if i < 9:
        y_test = np_label
    else:
        y_test = np_label_v
       
        
    #a = all_time_sort[i]
    y_score = all_time_sort[i]
    #    y_score = E_all_time
    #print(y_score)
    #print(y_test)
    y = np.vstack((y_score,y_test))
    y_ = y.T[np.lexsort(-y[::-1,:])].T
    y = y_[:,0:k]
    
    y_test = y[1,:].reshape((-1,1))      #实际停车的标签1停车0没停车
    y_score = y[0,:].reshape((-1,1))     #我们预测是可能可能停车的概率
    
    fpr, tpr,thrsh = roc_curve(y_test,y_score, pos_label=None, sample_weight=None, drop_intermediate=True)
            
    AUC = auc(fpr,tpr)
    methods = ['W', 'E', 'R', 'V']
    indicators = ['AST', 'MST', 'TAT']
    if (i) % 1 == 0:
        i_i = int(i % 3)
        m_i = int(i / 3)
        print('==========' + methods[m_i]+'-'+indicators[i_i] + '==========')
        print('AUC:' + str(AUC))
        
        #其中：y_test为测试集的真实label, y_score为模型预测得到的Score值
        # precision, recall, thresholds = precision_recall_curve(y_test,y_score)

        # 计算精确率和召回率
        n_thresholds = len(np.unique(y_score))
        precision, recall, thresholds = precision_recall_curve(y_test,y_score)
        # 计算AP
        ap = metrics.auc(recall, precision)
        #print('AP:', ap)

        precision1, recall1, thresholds1 = PR_thresh(y_test,y_score)
        area = metrics.auc(recall1, precision1)
        #print ('AP area :' + str(area))
        #precision.sort()
        #plt.title('Precision/Recall Curve')
        #plt.xlabel('Recall')
        #plt.ylabel('Precision')
        #plt.plot(recall,precision)
    #    recall.sort(reverse = True)
        step_kwargs = ({'step':'post' } if 'step' in signature(plt.fill_between).parameters else {})
        
        
        plt.step(recall, precision, color='b', alpha=0.2,where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])

        #plt.show()

        APs = average_precision_score(y_test,y_score)
        
        print('APs:' + str(APs))
        
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw,marker='o',label='ROC curve (AUC = %0.4f)' %AUC)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--',marker='o')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
    
#result.append([k,AUC,APs])
        

        