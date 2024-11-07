%SVD-Thresholding
function T = SVDThreshold(x,alpha)
%=====our version======
[U,Sigma,V] = svd(x,'econ');
Sigma = max(0,Sigma - alpha);

T = U*Sigma*V';

%=====lrsd version=====
% [U,D,V] = svd(x,'econ');
% V = V';
% D = diag(D);
% ind = find(D > alpha);
% D = diag(D(ind) - alpha);
% T = U(:,ind) * D * V(ind,:);
