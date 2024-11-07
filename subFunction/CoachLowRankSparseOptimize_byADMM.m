function [I,E,theta,Loss,rho] = CoachLowRankSparseOptimize_byADMM(R,I_original,para)
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
% 
% 
%=============���ֲ�����ʼ��==============
isPrint = 1;
tol = 1e-6; 
maxIter = 200;

if isfield(para,'lambda'); lambda = para.lambda; end
if isfield(para,'rho'); rho = para.rho; end
if isfield(para,'maxIter'); maxIter = para.maxIter; end
if isfield(para,'tol'); tol = para.tol; end
if isfield(para,'isPrint'); isPrint = para.isPrint; end

[N,M] = size(R);

%ԭʼI�����й̶�Ϊ1��idx
% [row_fixed,col_fixed]= find(I_original > 0);
idx_fixed = find(I_original >0);

%��ʼ��
E = R./50.0;
I = I_original;
theta =  R./50.0;
Y_1 = ones(N,M);
Y_2 = ones(N,M);
% loss = [];
norm_E = [];
norm_I = [];
constraint1 = [];
constraint2 = [];

%��������
isConverge = 0;
iter = 0;
while ~isConverge
    iter = iter + 1;
    
    %������������
    E_new =  OptimizeE_ByST(R,I,Y_1,lambda,rho);
    theta_new = OptimizeTheta_BySVDt(R,I,Y_2,rho);
    I_new = OptimizeI(R,E_new,theta_new,Y_1,Y_2,rho);
    I_new(idx_fixed) = 1.0;
    
    
    Y_1_new = Y_1 + rho*(R - R.*I_new - E_new  );
    Y_2_new = Y_2 + rho*(theta_new - R.*I_new );
    
    %�������ε����ı仯��                    
    norm_E(iter) = norm(E_new - E,'fro');
    norm_I(iter) = norm(I_new - I,'fro');
    norm_theta(iter) = norm(theta_new - theta,'fro');
%     loss(iter) = rank(theta_new)+ lambda*norm(E_new,1);
    constraint1(iter) = norm(R - R.*I_new - E_new,'fro');
    constraint2(iter) = norm(theta_new - R.*I_new,'fro');
    
    %�Ƿ���ʾϸ��
    if isPrint == 1
        disp(['iter' num2str(iter) ' ...']);
        disp(['...norm_E:' num2str(norm_E(iter))]);
        disp(['...norm_I:' num2str(norm_I(iter))]);
        disp(['...norm_theta:' num2str(norm_theta(iter))]);
    %     disp(['......loss:' num2str(loss(iter))]);
    end 
    
    %�ж��Ƿ�����
    if (norm_E(iter) < tol) && (norm_I(iter) <tol) && (norm_theta(iter) <tol)
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
    Y_1 = Y_1_new;
    Y_2 = Y_2_new;
end
%����������loss�͸������仯������Ϊstruct
% Loss.loss = loss;
Loss.norm_E = norm_E;
Loss.norm_I = norm_I;
Loss.norm_theta = norm_theta;
Loss.constraint1 = constraint1;
Loss.constraint2 = constraint2;


%ʹ��Soft-Thresholding�������Ż�����E
function  E_new =  OptimizeE_ByST(R,I,Y_1,lambda,rho)
alpha = lambda*1.0/rho;
x = R + Y_1*1.0/rho + R.*I;
E_new = softThreshold(x,alpha);
E_new(E_new <0) = 0;


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





%%======ʹ��Projected Gradient Decend�������Ż�����I
%%======(���ã�����I�в����ھ������㣬���������ݶ��½�����ֻ��Ҫʹ�ö��κ�������ֵ��ʽ����)
% function I_new = OptimizeI_ByPGD(R,I,E,theta,Y_1,Y_2,rho,gamma)
% %step1 �ݶ��½���
% I_grad = getGradientOfI(R,I,E,theta,Y_1,Y_2,rho);
% I_new = I - gamma.*I_grad;
% 
% %step2 f��Χӳ��
% I_new(I_new > 1) = 1;
% I_new(I_new < 0) = 0;




