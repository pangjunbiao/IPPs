clear ;close all;
%%
addpath('./subFunction/');

%%
rand('seed',0);
R = rand(100,50); %N-by-M�ľ���,����NΪ�켣����MΪĳ��·��������ΪM������
% R_temp = getKNNGraph(R,10);
% R = full(R_temp);

para = [];
para.lambda = 0.1; %sparseԼ���ķ���������
para.rho = 0.2;    %ADMM����
para.maxIter = 100;  %����������,default = 200
para.tol = 1e-6;
para.isPrint = 0;

Loss = [];

version = 'ADMM';
disp(version)
if strcmpi(version,'ADMM')
    %===================ADMM================
    [I,E,theta,Loss] = CoachLowRankSparseOptimize_byADMM(R,para);
    Loss.LowRank = theta;
    Loss.Sparse = E;
 
elseif strcmpi(version,'LRSD')
    %===================LRSD================
    [m,n] = size(R);
    opts = [];
    opts.beta = 0.10;
    opts.tol = 1e-6;
    opts.maxit = 1000;
    opts.A0 = zeros(m,n);
    opts.B0 = zeros(m,n);
    opts.Lam0 = zeros(m,n);
    opts.record_res = 1;
    t = .1;
    out = lrsd(R, t/(1-t), opts);
    Loss.constraint = out.res;
    Loss.LowRank = out.LowRank;
    Loss.Sparse = out.Sparse;
end

%% ��ͼ

if strcmpi(version,'ADMM')
    figure
    h1 = plot(Loss.constraint1);
    title('constraint1')
    figure
    h2 = plot(Loss.constraint2);
    title('constraint2')
elseif strcmpi(version,'LRSD')
    figure
    h3 = plot(Loss.constraint);
    title('constraint')
end


disp(['Rank of theta:' num2str(rank(Loss.LowRank))]);
disp(['Sparse of E:' num2str(length(find(Loss.Sparse == 0))*1.0/numel(Loss.Sparse))]);