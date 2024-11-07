main_LowRank_RI:
Main Program
The main program input data is located in ./cluster_12_3-12, which is obtained from the 2-tree data/3 data preprocessing program.
data_matrix:
Results from the existing data run.
Auc+ROC+AP.py:
Auc and AP result testing program.

1. Program execution steps are obtained from the matlab program man_LowRank_RI.m to get W_time.mat, E_time.mat, R_time.mat, V_time.mat.

2. Run the geBoundOfClusters.py program to obtain the corresponding interpolated distance data, result matrix, and result matrix_ex.

3. Run the Auc+ROC+AP.py to obtain the predicted data.
