main_LowRank_RI:
主程序
主程序输入数据在./cluster_12_3-12，该数据由2树数据/3数据预处理的程序中获得
data_matrix：
现有数据跑出的结果
Auc+ROC+AP.py：
Auc、AP结果测试程序


1 程序运行步骤由matlab程序man_LowRamnk_RI.m获得W_time.mat、E_time.mat、R_time.mat、V_time.mat

2 运行geBoundOfClusters.py程序获得相应的距离的插值数据，result matrix  and result matrix_ex

3 运行Auc+ROC+AP.py获得预测数据
