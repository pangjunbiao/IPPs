%Soft-Thresholding
function S = softThreshold(x,alpha)
S = max(x-alpha, 0) + min(x+alpha, 0);

