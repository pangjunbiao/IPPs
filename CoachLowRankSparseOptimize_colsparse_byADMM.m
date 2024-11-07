function [I,E,theta,W,Loss,rho] = CoachLowRankSparseOptimize_colsparse_byADMM(R,I_original,para)

%ʹ��ADMM�������������Ż�����ϡ��Լ��
%
%INPUT:
%   R:      N-by-M�ľ���,����NΪ�켣����MΪĳ��·��������ΪM������
%   I_original: ����ͣ��ģ�ͣ��ж�Ϊ���Ƕ���ͣ����ĵ㡣 
%   lambda:     sparseԼ���ķ���������
%   rho:        ADMM����
%   maxIter:    ����������,default = 200
%   tol:        ����ֹͣ�ľ���.default = 1e-6 
%   isPrint:    �����Ƿ�printÿ��ѭ���е�ϸ����Ϣ
% OUTPUT:
%   I:          N-by-M��LowRank�������У�0<=I_{ij}<=1����ʾ��͵�����ͣ����
%   E:          N-by-M��sparse�������У�E_{ij}>=0����ʾ��͵ķ�����ͣ���㣬�����͵㡣
%   W��         N-by-M��column sparse�������У�W_{ij}>=0����ʾ��͵ķ�����ͣ���㣬�����͵㡣
% 
%=============���ֲ�����ʼ��==============
isPrint = 1;
tol = 1e-6; 
maxIter = 500;

if isfield(para,'lambda'); lambda = para.lambda; end
if isfield(para,'rho'); rho = para.rho; end
if isfield(para,'maxIter'); maxIter = para.maxIter; end
if isfield(para,'tol'); tol = para.tol; end
if isfield(para,'isPrint'); isPrint = para.isPrint; end
if isfield(para,'beta'); beta = para.beta; end
if isfield(para,'u'); u = para.u;end

[N,M] = size(R);

%ԭʼI�����й̶�Ϊ1��idx
% [row_fixed,col_fixed]= find(I_original > 0);
idx_fixed = find(I_original >0);

%��ʼ��
E = R./50.0;
I = I_original;
theta =  R./50.0;
W =  R./50.0;
% W =  zeros(N,M);
Y_1 = ones(N,M);
Y_2 = ones(N,M);
Y_3 = ones(N,M);
% loss = [];
norm_E = [];
norm_I = [];
norm_W = [];
constraint1 = [];
constraint2 = [];
constraint3 = [];

%��������
isConverge = 0;
iter = 0;
while ~isConverge
    iter = iter + 1;

    if iter ==17
        
       a = 1;
    end

    %������������
    if iter == 1
        E_new =  OptimizeE_ByST(R,I,W,Y_1,Y_3,lambda,rho);
    else
        E_new =  OptimizeE_ByST(R,I,W_new,Y_1,Y_3,lambda,rho);

    end
%     E_new =  OptimizeE_ByST(R,I,W_new,Y_1,Y_3,lambda,rho);
%     E_new =  OptimizeE_ByST(R,I,Y_1,lambda,rho);
    theta_new = OptimizeTheta_BySVDt(R,I,Y_2,rho);
    I_new = OptimizeI(R,E_new,theta_new,Y_1,Y_2,rho);
    I_new(idx_fixed) = 1.0;
    W_new = OptimizeW_ByST(R,E_new,Y_3,beta,rho);
    

    
    Y_1_new = Y_1 + rho*(R - R.*I_new - E_new  );
    Y_2_new = Y_2 + rho*(theta_new - R.*I_new );

    Y_3_new = Y_3 + rho*(W_new - E_new );

    rho_new = rho*u;

    
    %�������ε����ı仯��                    
    norm_E(iter) = norm(E_new - E,'fro');
    norm_I(iter) = norm(I_new - I,'fro');
    norm_theta(iter) = norm(theta_new - theta,'fro');
    norm_W(iter) = norm(W_new - W,'fro');

%     loss(iter) = rank(theta_new)+ lambda*norm(E_new,1);
    constraint1(iter) = norm(R - R.*I_new - E_new,'fro');
    constraint2(iter) = norm(theta_new - R.*I_new,'fro');
    constraint3(iter) = norm(W_new - E_new,'fro');

    
    %�Ƿ���ʾϸ��
    if isPrint == 1
        disp(['iter' num2str(iter) ' ...']);
        disp(['...norm_E:' num2str(norm_E(iter))]);
        disp(['...norm_I:' num2str(norm_I(iter))]);
        disp(['...norm_theta:' num2str(norm_theta(iter))]);
        disp(['...norm_W:' num2str(norm_W(iter))]);
    %     disp(['......loss:' num2str(loss(iter))]);
    end 
    
    %�ж��Ƿ�����
    if (norm_E(iter) < tol) && (norm_I(iter) <tol) && (norm_theta(iter) <tol) && (norm_W(iter) <tol)
        isConverge = 1;
        disp(['reach the tol:' num2str(tol)])

    elseif iter == maxIter
        isConverge =1;
        disp(['reach the maxIter:' num2str(maxIter) ])
    end
    
    %���µ���ֵ
    E = E_new;
    I = I_new;
    theta = theta_new;
    W = W_new;
    Y_1 = Y_1_new;
    Y_2 = Y_2_new;
    Y_3 = Y_3_new;
    rho = rho_new;
end 
%����������loss�͸������仯������Ϊstruct
% Loss.loss = loss;
Loss.norm_E = norm_E;
Loss.norm_I = norm_I;
Loss.norm_theta = norm_theta;
Loss.norm_W = norm_W;
Loss.constraint1 = constraint1;
Loss.constraint2 = constraint2;
Loss.constraint3 = constraint3;


%ʹ��Soft-Thresholding�������Ż�����E
function  E_new =  OptimizeE_ByST(R,I,W,Y_1,Y_3,lambda,rho)
alpha = lambda*1.0/(2*rho);

% for i = 1:size(E,1)
%     norm_i = norm(E(i,:),2);
%     E_(i,i) =  norm_i;
% end
% E_ = E_ * E;

x = 0.5 * (R + W + Y_1*1.0/rho + Y_3*1.0/rho - R.*I);
E_new = softThreshold(x,alpha);
E_new(E_new <0) = 0;

% function  E_new =  OptimizeE_ByST(R,I,Y_1,lambda,rho)
% alpha = lambda*1.0/rho;
% x = R + Y_1*1.0/rho - R.*I;
% E_new = softThreshold(x,alpha);
% E_new(E_new <0) = 0;

function W_new = OptimizeW_ByST(R,E,Y_3,beta,rho)
alpha = beta*1.0/(rho);
x =(E - Y_3*1.0/rho);
% x = (R + Y_3*1.0/rho - R.*I );

[N,M] = size(R);
for i = 1:M
    norm_i = norm(x(:,i),1);
    if alpha < norm_i
        u = (norm_i-alpha)*1.0/norm_i;
        W_new(:,i) = u * x(:,i);
    else
        W_new(:,i) = zeros(N,1);
    end
end
% if size(W_new ,1) < N
%     W_new = zeros(N,M);
% end
W_new(W_new <0) = 0;




%ʹ��SVD-Thresholding�������Ż�����theta
function theta_new = OptimizeTheta_BySVDt(R,I,Y_2,rho)
x = R.*I - Y_2*1.0/rho;
alpha = 1.0/rho;

theta_new = SVDThreshold(x,alpha);
theta_new(theta_new <0) = 0;


%ʹ�ö��κ�������ֵ��ʽ������Ż�����I
function I_new = OptimizeI(R,E,theta,Y_1,Y_2,rho)
% step1 ʹ�ù�ʽ�������ֵ��
para_a = rho * R;
para_b = rho.*E - rho.*theta - rho.*R - Y_1 - Y_2;
I_new  = zeros(size(R));
I_new(R > 0) = - (para_b(R > 0)*1.0./(2*para_a(R > 0)));

% step2 �������ʵ㷶Χӳ�䵽[0,1]��
I_new(I_new > 1) = 1;
I_new(I_new < 0) = 0;







