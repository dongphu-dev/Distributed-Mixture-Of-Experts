function [sumdis, KLdisvec] = KL_distance(expert1, expert2, X_val)      

    %---------------------------------------------------------------
    %Compute conditional KL distance between two Gaussian regression
    
    [n, ~] = size(X_val);
    mean1 = X_val * expert1.xBeta;
    mean2 = X_val * expert2.xBeta;

    KLdisvec =  1/2*(  log(expert2.sigma2 / expert1.sigma2)...
                + expert1.sigma2 / expert2.sigma2...
                + (mean2-mean1).^2/expert2.sigma2...
                - 1);
    sumdis = sum(KLdisvec);
    %---------------------------------------------------------------
    
end