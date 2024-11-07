clear ;close all;
%%
addpath('./subFunction/');

%%
rand('seed',0);
R = rand(100,50); %N-by-M的矩阵,其中N为轨迹数，M为某段路径被划分为M个区域
% R_temp = getKNNGraph(R,10);
% R = full(R_temp);

para = [];
para.lambda = 0.1; %sparse约束的罚函数参数
para.rho = 0.2;    %ADMM参数
para.maxIter = 100;  %最大迭代次数,default = 200
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

%% 作图

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