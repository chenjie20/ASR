function [Zs] = simple_graph_fusion(Wn)
%----Wn: 1 * nv -----------------------
%-----nv represents the number of views
%----Wn{index}: n * n
%---------------------------------------

nv = size(Wn, 2);
n = size(Wn{1}, 1);
Zs = zeros(n, n);
for i = 1 :n
    for j =  1: n
        z = 0;
        count = 0;
        for nv_idx = 1 : nv
            current_value = abs(Wn{nv_idx}(i, j));
            if current_value > 1e-6
                count = count + 1;
                z = z + current_value;
            end                        
        end
        if count > 1
            z = z / count;
        end
        Zs(i, j) = z;
    end
end

end