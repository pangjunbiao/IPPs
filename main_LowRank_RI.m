                          clear ;close all;
%%
addpath('./subFunction/');
%% ����˵��
%E_num : ������ʱ��ͳ��ÿ�������ͣ��������
%E_ave : ������ʱ��ͳ��ÿ�������ͣ��ʱ��
%idxe_all : ���ఴ��ÿ��ͣ��������ܺ�������
%idxe_max : ���ఴ��ÿ��ͣ����������ֵ������
%idxe_all : ���ఴ��ÿ��ͣ�������ǰ2�����ֵ��������

%% ���ݶ���
meter_para =200;    %
threshold = 1; % define pick up point

%% ������

DataRoot = 'cluster_12_3-12/';
DataRoot_matrix = ['cluster_12_3-12/result matrix/matrix_meter' num2str(meter_para) '/'];   %V矩阵要在matrix后面加_ex
DataRootex_matrix = ['cluster_12_3-12/result matrix_ex/matrix_meter' num2str(meter_para) '/'];

    
% index_cluster = 15;    %old_3

E_num_all_ = zeros(23,2);
E_num_max_ = zeros(23);
E_num_col_= zeros(23);
E_num_time = zeros(9,2);
car_time = [6,7,8,9,10,11,12,13,18];
E_num_time(:,1) = car_time;
E_num_all_cluster = []
E_num_max_cluster = []
E_num_col_cluster = []
tk = 2

% E_all = [];
% W_all = [];
% E_an_all = [];
%for index_cluster = [12]
%for index_cluster = [1,2,3,4,5,6,8,9,11]  %for paper cluster

repet_and_out_5_cluster = [7, 11, 17, 19, 20, 0, 2, 3, 4, 5]
repet_and_out_5_cluster_ex = [7, 11, 12, 14, 1, 6, 8, 14, 16, 19]  %v矩阵时用
for index_cluster = 1:23    %v矩阵时22
    if ismember(index_cluster, repet_and_out_5_cluster) ==1 
        continue;
    end
    R_path = [DataRoot_matrix 'R/R_matrix_of_cluster' num2str(index_cluster) '.txt'];
    R_old = load(R_path);

    ex_path = [DataRoot_matrix 'R/R_matrix_of_cluster' num2str(index_cluster) '.txt'];  %ex
    ex = load(ex_path);

    
    R_path = [DataRoot_matrix 'R/R_matrix_of_cluster' num2str(index_cluster) '.txt'];  %ex
    I_path = [DataRoot_matrix 'I/I_matrix_of_cluster' num2str(index_cluster) '.txt'];   %ex 
    info_path = [DataRoot_matrix 'info/info_matrix_of_cluster' num2str(index_cluster) '.txt'];  %ex
    startTime_path = [DataRoot 'start_time/' num2str(index_cluster) '.txt'];   %v矩阵时加time后加_ex
    
    R = load(R_path);
    I_original = load(I_path);
    disp(['Sparse of R:' num2str(length(find(R == 0))*1.0/numel(R))]);
    disp(['Rank of R:' num2str(rank(R))]);

    start_time = load(startTime_path);
    arv_time = start_time(:,6);

    start_time = start_time(:,1:3);
    start_time(:,3) = mod(start_time(:,3),10000);
    start_time(:,4) = floor(start_time(:,3)/100);
    start_time(:,5) = mod(start_time(:,3),100);
    
    start_time(:,7) = floor(arv_time/100);
    start_time(:,8) = floor(start_time(:,7)/100);
    start_time(:,9) = mod(start_time(:,7),100);

    
    info = load(info_path);
   
 

    %% startTime ����
    [~,idx_sort] = sort(start_time(:,3));
    start_time_sort = start_time(idx_sort,:);
    ST_sort_cell = num2cell(start_time);
    for Irow = 1:size(ST_sort_cell,1)
        ST_sort_cell{Irow,3} = num2str(ST_sort_cell{Irow,3});
        ST_sort_cell{Irow,4} = ST_sort_cell{Irow,3}(9:end);
        ST_sort_cell{Irow,5} = str2num(ST_sort_cell{Irow,4});
    end
    [~,idx_sort] = sort(cell2mat(ST_sort_cell(:,5)));
    ST_sort_cell = ST_sort_cell(idx_sort,:);

    %% pre-processing
    % R_rescale = (R - min(min(R)))./(max(max(R)) - min(min(R)));%better
    R_rescale = R;
    %% �����趨
    para = [];
    para.lambda = 0.1; %sparseԼ���ķ���������
    para.rho = 1;    %ADMM����
    para.maxIter = 200;  %����������,default = 237
    para.tol = 1e-7;
    para.isPrint = 0;
    para.beta = 0.1;  %betaֵԽС��������E�Ƿ��Խ�࣬����ֵԽ�ȶ���������
    para.u=1.2;

    opt_compare = 'sum';
    %===================ADMM================
    [I_an,E_an,theta_an,Loss_an,rho] = CoachLowRankSparseOptimize_byADMM(R_rescale,I_original,para);      %δ����column sparse
    W = zeros(164,24);
    [I,E,theta,W,Loss,rho] = CoachLowRankSparseOptimize_colsparse_byADMM(R_rescale,I_original,para);       %������column sparse
    [I,E2,theta,W2,Loss1,rho] = CoachLowRankSparseOptimize_colsparse2_byADMM(R_rescale,I_original,para);
    %plot(Loss.norm_W(1:50));
    a = norm(W-E_an);
    %xlabel('epoch');
    %ylabel('loss/s');
    disp(['cluster:' num2str(index_cluster) ' E-W norm:'  num2str(a) ])
    


    Loss.LowRank = theta;
    Loss.Sparse = E;
    Loss.colsparse = W;
    
  



    %% �о�E����
    start_time(:,6) = round(start_time(:,4)+start_time(:,5)./60.0);
    start_time_unique = unique(start_time(:,6));
    
    start_time(:,10) = round(start_time(:,8)+start_time(:,9)./60.0);
    arv_time_unique = unique(start_time(:,10));    


    %% ���ճ���ʱ������������
    E= W;   %E矩阵 E_an   R矩阵直接用R表示   W is our matrix   V也是用的W矩阵
    [m,n] = size(E);
    
    E_num = zeros(length(start_time_unique),n+1);
    for i = 1:length(start_time_unique)
        st_idx = find(start_time(:,6) == start_time_unique(i));
        E_unique = E(st_idx,:);
        E_num(i,1)= start_time_unique(i);
        for j=1:n
            a = find(E_unique(:,j)>threshold);
            num = length(a);
            E_num(i,j+1) = num;
            % if j == 19 && i ==5
            %     a = 0;
            % end

        end
    end
    E_num(:,3:n+2) = E_num(:,2:n+1);
    E_num(:,n+3) = sum(E_num(:,3:n+2),2);
    E_num(:,n+4) = E_num(:,n+3)*1.0/sum(E_num(:,n+3));



    %% ����E����С��10s���޳�
    [m,n] = size(E);
    E_rem = zeros(m,n);
    % E_rem(E>threshold) = E(E>threshold);
    for i=1:m
        for j=1:n
            aa = E(i,j);
        if aa>threshold
            E_rem(i,j) = aa;
        end
        end
    end

    % E_not0 = sum(E_rem>0,2);


    %% ����ƽ��ʱ��
    [m,n] = size(E);
    E_plate = cell(length(start_time_unique),3);
    E_ave = zeros(length(start_time_unique),n+1);
    for i = 1:length(start_time_unique)
        st_idx_ = find(start_time(:,6) == start_time_unique(i));
        plate = start_time(st_idx_,1);
        date = start_time(st_idx_,2);
        stop_point_all = zeros(length(plate),1);
        % stop_data = E_rem(st_idx_,:);
        % stop_point = sum(stop_data,1);
        % E_plate(i,4) = {[plate,date,stop_point]};
        E_plate(i,1) = {start_time_unique(i)};
        % E_plate(i,2) = {[plate,date]};
        % E_plate(i,) = {date(1:end)};
        E_plate(i,3) = {length(plate)};
        stop_plate = unique(plate);
        stop_point = zeros(length(stop_plate),n);
        stop_num = zeros(length(stop_plate),1);
        for ii = 1:length(stop_plate)
            s_plate = stop_plate(ii);
            stop_idx = find(plate == s_plate);
            stop_data = E_rem(st_idx_(stop_idx),:);
            stop_data(stop_data>0) = 1;
            stop_point(ii,:) = sum(stop_data,1);
            stop_num(ii) = sum(stop_point(ii,:));
        end
            E_plate(i,4) = {[stop_plate,stop_num,stop_point]};

        for num=1:length(plate)
            stop_point_num = find(E_rem(st_idx_(num),:) > threshold);
            stop_point_all(num,1) = length(stop_point_num);
        end 
        E_plate(i,2) = {[plate,date,stop_point_all]};
        E_ave(i,1) = start_time_unique(i);
        [row, col] = find( E_rem(st_idx_,:) ~= 0 );
        E_rem_row = unique(row);
        E_rem_row_ = st_idx_(E_rem_row,:);
        if ~isempty(E_rem_row)
            E_ave_ = E_rem(E_rem_row_,:);
            if i ==5
                a = 0;
            end
            E_ave(i,2:n+1) = mean(E_ave_,1);
        end
    end

    E_ave_final = zeros(length(start_time_unique),n+1);
    for i = 1:length(start_time_unique)
        st_idx = find(start_time(:,6) == start_time_unique(i));
        temp = E_rem(st_idx,:);
        E_ave_final(i,1)= start_time_unique(i);
        for j =1:n
            E_ave_final(i,j) = sum(temp(:,j))*1.0 ;
            
            % E_ave_final(i,j) = sum(temp(:,j))*1.0/(sum(temp(:,j) > threshold)+eps);
        end
    end



    E_ave_final(:,3:n+2) = E_ave_final(:,1:n);
    E_ave_final(:,1) = E_num(:,1);
    E_ave_final(:,2) = cell2mat(E_plate(:,3));
    E_num(:,2) = cell2mat(E_plate(:,3));

    
    %每列总和
    E_num_all_ = sum(E_num(:,3:n+2),1);%(m+1,3:n+2)(m+1,3:n+2)
    E_num_all_cluster = [E_num_all_cluster,E_num_all_];

    %每列最大值
    E_num_max_ = max(E_num(:,3:n+2),[],1);
    E_num_max_cluster = [E_num_max_cluster,E_num_max_];

    %每列前k个平均
    E_num_col = E_num(:,3:n+2);
    E_num_col_sort_ = sort(E_num_col,'descend');
    E_num_col_ = sum(E_num_col_sort_(1:tk,:),1)/tk;
    E_num_col_cluster = [E_num_col_cluster,E_num_col_];




end
[E_num_all_sort,idxe_all] = sort(E_num_all_(:,2),'descend');
[E_num_max_sort,idxe_max] = sort(E_num_max_(:,1),'descend');
[E_num_col_sort,idxe_col] = sort(E_num_col_(:,1),'descend');
time_s = start_time(:,4)*60 + start_time(:,5);
time_a = start_time(:,8)*60 + start_time(:,9);
start_time(:,11) = time_a-time_s;


%根据所需矩阵更改得到 W 、E、 R、 V
save W_time.mat E_num_all_cluster E_num_max_cluster E_num_col_cluster  

