function out = log_softmax(X, W)
%
K = size(W,2);

XW = X*W;
Z = sum(exp(XW), 2);
out = XW - repmat(log(Z), 1, K);
end
