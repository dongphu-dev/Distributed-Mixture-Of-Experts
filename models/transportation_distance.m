function tran_dis = transportation_distance(large_mixture, reduced_mixture, plan, X_val)

    [n, d] = size(X_val);
    KM  = length(large_mixture.variances);
    K   = length(reduced_mixture.variances);
    tran_dis = [];
    

    for l=1:KM
        
        expert1.xBeta  = large_mixture.experts(:,l);
        expert1.sigma2 = large_mixture.variances(l);
        
        for k=1:K
            
            expert2.xBeta =  reduced_mixture.experts(:,k);
            expert2.sigma2 = reduced_mixture.variances(k);
            
            [~, KLdisvec] = KL_distance(expert1, expert2, X_val);
            
            if length(plan{l,k}) > 1
                expXA  = exp(X_val*reshape(plan{l,k}, d, K));
                weight = expXA ./ sum(expXA,2);
                weight = weight(:,k);
            else
                weight = 0;
            end
            
            KLdisvec = weight .* KLdisvec;
            tran_dis = [tran_dis sum(KLdisvec)];
            
        end
    end
    
    tran_dis = sum(tran_dis);

end