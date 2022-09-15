function [Wn, iter, max_errors] = sacr(data_views, Mn, alpha, beta, dim)
% ---------------------------------------------------------------------------------------------
%---data_views: nv * m * n
%---nv represents the number of views
%----m and n represent the dimensionality and the number of the features, respectively.
%----Mn: 1 * nv, consists of nv cells. Each cell contains a diag matrix of size n * n.
%----Each diagonal element of Mn may be 0 or 1. 
%----0 indicates that the corresponding feature is missing.
%---------------------------------------------------------------------------------------------

    rho = 1.1;
    max_mu = 1e10;
    mu = 1e-2;
    maxIter = 500;
    tol = 1e-6;
    
    max_errors = zeros(1, maxIter);
    
    nv = length(data_views);
    Xn = cell(1, nv);
    Dn = cell(1, nv);
    Zn = cell(1, nv);
    Jn = cell(1, nv);
    Yn = cell(1, nv);
    Wn = cell(1, nv);
    cols = cell(1, nv);
    
    pous = 1 / (nv - 1) * ones(nv, 1);    

    num_samples = zeros(nv, 1);
    errors = zeros(nv, 1);
    for idx = 1 : nv
        cols{idx} = find(abs(diag(Mn{idx}) - 1) < 1e-6);
        Xn{idx} = data_views{idx}(:, cols{idx});
        [m, n] = size(Xn{idx});
        if dim > 0 && dim < m
            for i = 1 : n
                sample = Xn{idx}(:, i);
                Xn{idx}(:, i) = sample ./ max(1e-12, norm(sample));
            end 
            [eigen_vector, ~] = f_pca(Xn{idx}, dim);
            Xn{idx} = eigen_vector' *  Xn{idx};        
        end
        Xn{idx} = normc(Xn{idx});
        
        num_samples(idx, 1) = n;
        Dn{idx} = Xn{idx};
        Zn{idx} = zeros(n, n);
        Jn{idx} = zeros(n, n);
        Yn{idx} = zeros(n, n);
        
        nn = size(data_views{idx}, 2);
        Wn{idx} = zeros(nn, nn);
    end
      
    iter = 0;
    while iter < maxIter
        for idx = 1 : nv            
            n = num_samples(idx, 1);
            
            % update J{idx}
            Z_sum = zeros(n, n);
            for j = 1 : nv
                if idx ~= j
                    Z_sum = Z_sum + Wn{idx}(cols{idx}, cols{idx});
                end
            end
            tmp1 = (Dn{idx}' * Dn{idx}) + (mu + beta) * eye(n);
%             tmp2 = (Dn{idx}' * Xn{idx}) + mu * Zn{idx} + beta * (1 / (nv - 1) * Z_sum) - Yn{idx};  
            tmp2 = (Dn{idx}' * Xn{idx}) + mu * Zn{idx} + beta * pous(idx) * Z_sum - Yn{idx};   
            Jn{idx} = normc(tmp1 \ tmp2);
            
            % update Z{idx}
            tmp = Jn{idx} + Yn{idx}/mu;
            thr = sqrt(1 * alpha/mu);
            Zn{idx} = tmp.*((sign(abs(tmp)-thr)+1)/2);
            ind = abs(abs(tmp)-thr) <= 1e-6;
            Zn{idx}(ind) = 0;
            Zn{idx} = Zn{idx} - diag(diag(Zn{idx}));
            
%             Zn{idx} = max(0, abs(tmp) - alpha/mu * ones(n)) .* sign(tmp);    
%             Zn{idx} = Zn{idx} - diag(diag(Zn{idx}));

            Wn{idx}(:, :) = 0;
            Wn{idx}(cols{idx}, cols{idx}) = Zn{idx};  

             % update D{idx}
            A = Zn{idx} * Zn{idx}'; 
            B = Xn{idx} * Zn{idx}';
%             A = Jn{idx} * Jn{idx}'; 
%             B = Xn{idx} * Jn{idx}';
            for i = 1 : n
                if(A(i, i) ~= 0)
                    a = 1.0 / A(i,i) * (B(:,i) - Dn{idx} * A(:, i)) + Dn{idx}(:,i);
                    Dn{idx}(:,i) = a / (max( norm(a, 2),1));		
                end
            end
            
             % update an Lagrange multiplier
            Yn{idx} = Yn{idx} + mu * (Jn{idx} - Zn{idx});
            
            % update a penalty parameter
            mu = min(rho * mu, max_mu);
            
            errors(idx) = max(max(abs(Jn{idx} - Zn{idx}))); 
        end
        
        %update pou
        for i = 1 : nv
           pous(i) = 1 / (2 * norm(Jn{i} - Wn{i}(cols{i}, cols{i}), 'fro')); 
        end
        
        iter = iter + 1;
        finish = 0;
        max_error = max(errors);
        max_errors(1, iter) = max_error;
        if max_error < tol
            finish =  1;
        end
        if (iter == 1 || mod(iter, 50) == 0 || finish == 1)
            for idx = 1 : nv 
                disp(['iter ' num2str(iter) ', nv=' num2str(idx) ', mu=' num2str(mu, '%2.1e') ...
                     ', min value=' num2str(errors(idx),'%2.3e')]);
            end
        end       
        if finish == 1
            disp('srasf done.');
            break;
        end 
    end
end

