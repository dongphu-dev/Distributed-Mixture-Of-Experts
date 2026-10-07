function out = MSE(A,B)

    % Compute mean square error

    if size(A,1)<size(A,2), A = A';end
    if size(B,1)<size(B,2), B = B';end

    n1 = size(A,1);
    n2 = size(B,1);
    
    if n1 ~= n2
        error('MSE error: A and B must have a same size');
    end

    out = norm(A-B)^2/n1;

end