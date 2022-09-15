function [W, iter, errors] = graph_fusion(Wn)
%----Wn: nv * n * n---------------------
%-----nv represents the number of views
%---------------------------------------
    maxIter = 500;
    tol = 5e-3;
    
    errors = zeros(1, maxIter);
    
    nv = length(Wn);
    pous = 1 / nv  * ones(nv, 1);
    n = size(Wn{1}, 2);
    W = zeros(n, n);
    W_sum = zeros(n, n);

    % initialize W
    for i = 1 : nv
        W = W + Wn{i};
    end
    W = 1 / nv * W;
    
    iter = 0;  
    obj_value = 0;
    while iter < maxIter
        iter = iter + 1;
        obj_value_tmp = obj_value;
        obj_value = 0;
        
        %update pou
        for i = 1 : nv
           pous(i) = 1 / (2 * norm(W - Wn{i}, 'fro')); 
        end

         % update W
        for i = 1 : nv
            W_sum = W_sum + pous(i) * Wn{i};
        end 
        [W, ~] = project_simplex((W_sum));     
        W = W - diag(diag(W));

        % cacluate the objective vlaue
        for i = 1 : nv
            obj_value = obj_value + pous(i) * norm((W - Wn{i}), 'fro');
        end
        rr = abs((obj_value - obj_value_tmp) / obj_value);
        errors(1, iter) = rr;
        if (iter == 1 || mod(iter, 50) == 0)
            disp([iter, rr]);
        end        
        if(rr < tol)
            disp([iter, rr]);
            break;
        end
    end    

end
