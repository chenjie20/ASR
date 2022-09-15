close all;
clear;
clc;

addpath('utility');

%---------------------- load data -------------------------
load('data/proteinFold_Kmatrix.mat');
n = length(Y);
nv = size(KH, 3);
K = length(unique(Y));
gnd = Y;

%---------------------data description-------------------------------------
%------nv represents the number of views-----------------------------------
%------Suppose the size of each cell is m * n, where m and n represent the 
% dimensionality and the number of the features, respectively. ------------
%--------------------------------------------------------------------------
data_views = cell(1, nv);
for nv_idx = 1 : nv
     data_views{nv_idx} = KH(:, :, nv_idx);
end

%---------- The missing ratios --------------------------------------------
%- There are four experiments. --------------------------------------------
%--For example-------------------------------------------------------------
%- 0 represents that all features are available.
%- 0.1 represents that 10% of features are randomly missing in each view.
%-------------------------------------------------------------------------
raitos = [0, 0.1, 0.3, 0.5];

%----------------The two parameters of SACR defined by users --------------
%-----Four groups of the parameters----------------------------------------
% For example, alphas(1) and betas(1) are used in the first experiment-----
%--------------------------------------------------------------------------
alphas = [1e-6, 1e-6, 1e-5, 5e-5];
betas = [0.07, 0.07, 0.05, 0.05];

%--We may a PCA algorithm to preprocess the original features of all samples
% a PCA algorithm will be skippend when the parameter is 0 ---------------
new_dims = [50, 50, 50, 50];

%----------- The following parameters remian unchanged ---------------------
sparsity_ratio = 0.2;
max_iters = 500;
errors2 = zeros(1, max_iters);
ratio_len = length(raitos);

testing_times = 10;
final_clustering_accs = zeros(ratio_len, testing_times);
final_clustering_nmis = zeros(ratio_len, testing_times);
final_clustering_purities = zeros(ratio_len, testing_times);
final_clustering_fmeasures = zeros(ratio_len, testing_times);
final_clustering_ris = zeros(ratio_len, testing_times);
final_clustering_aris = zeros(ratio_len, testing_times);
final_clustering_ratios =  zeros(ratio_len, testing_times);
final_clustering_iters = zeros(ratio_len, testing_times);
final_clustering_fusion_iters = zeros(ratio_len, testing_times);
final_clustering_costs = zeros(ratio_len, testing_times);
final_clustering_f_errors = zeros(ratio_len, testing_times, max_iters);
final_clustering_s_errors = zeros(ratio_len, testing_times, max_iters);
individual_view_sparsity_ratios = zeros(ratio_len, nv);
sparsity_ratios_bf_fusion = zeros(ratio_len, 1);
sparsity_ratios_af_fusion = zeros(ratio_len, 1);

Mn = cell(1, nv);
for raito_idx = 1 : length(raitos)
    stream = RandStream.getGlobalStream;
    reset(stream);
    raito = 1 - raitos(raito_idx);    
    rand('state', 100);
    for nv_idx = 1 : nv
        if raito < 1
             pos = randperm(n);
             num = floor(n * raito);
             sample_pos = zeros(1, n);
             sample_pos(pos(1 : num)) = 1;
             Mn{nv_idx} = diag(sample_pos);
        else
            Mn{nv_idx} = diag(ones(1, n));
        end
    end

    alpha = alphas(raito_idx);
    beta = betas(raito_idx);
    
    tic;
    [Wn, iter1, errors1] = sacr(data_views, Mn, alpha, beta, new_dims(raito_idx));
    for nv_idx = 1 : nv
        num_nonzeros = sum(sum(abs(Wn{nv_idx}) > 1e-6));
        individual_view_sparsity_ratios(raito_idx, nv_idx) = num_nonzeros / (n * n);
    end
    Zs = simple_graph_fusion(Wn);
    num_nonzeros = sum(sum(abs(Zs) > 1e-6));
    current_sparsity_ratio = num_nonzeros / (n * n);
    sparsity_ratios_bf_fusion(raito_idx, 1) = current_sparsity_ratio;
    if current_sparsity_ratio > sparsity_ratio
        [Zs, iter2, errors2] = graph_fusion(Wn);  
        num_nonzeros = sum(sum(abs(Zs) > 1e-6));
        sparsity_ratios_af_fusion(raito_idx, 1) = num_nonzeros / (n * n);
    end        
    W = (abs(Zs) + abs(Zs')) / 2;  
    time_cost = toc;
    for time_idx = 1 : testing_times
        [acc, nmi, purity, fmeasure, ri, ari] = calculate_clustering_results(W, gnd, K);
        final_clustering_accs(raito_idx, time_idx) = acc;
        final_clustering_nmis(raito_idx, time_idx) = nmi;
        final_clustering_purities(raito_idx, time_idx) = purity;
        final_clustering_fmeasures(raito_idx, time_idx) = fmeasure;
        final_clustering_ris(raito_idx, time_idx) = ri;
        final_clustering_aris(raito_idx, time_idx) = ari;
        final_clustering_iters(raito_idx, time_idx) = iter1;
        final_clustering_fusion_iters(raito_idx, time_idx) = iter2;
        final_clustering_costs(raito_idx, time_idx) = time_cost;
        final_clustering_f_errors(raito_idx, time_idx, 1:end) = errors1(1:end);
        final_clustering_s_errors(raito_idx, time_idx, 1:end) = errors2(1:end);
        disp([raito, time_idx, alpha, beta, acc, nmi, purity, fmeasure, ri, ari]);
        dlmwrite('results/protein_final_result.txt', [raito, time_idx, alpha, beta, acc, nmi, purity, fmeasure, ri, ari, time_cost, current_sparsity_ratio, iter1, iter2] , '-append', 'delimiter', '\t', 'newline', 'pc');
    end    
    dlmwrite('results/protein_final_average_result.txt', [raito, alpha, beta, mean(final_clustering_accs(raito_idx, :)), std(final_clustering_accs(raito_idx, :)), mean(final_clustering_nmis(raito_idx, :)), std(final_clustering_nmis(raito_idx, :)), ...
        mean(final_clustering_purities(raito_idx, :)), std(final_clustering_purities(raito_idx, :)), mean(final_clustering_fmeasures(raito_idx, :)), std(final_clustering_fmeasures(raito_idx, :)), mean(final_clustering_ris(raito_idx, :)), std(final_clustering_ris(raito_idx, :)), ... 
        mean(final_clustering_aris(raito_idx, :)), std(final_clustering_aris(raito_idx, :)), mean(final_clustering_iters(raito_idx, :)), mean(final_clustering_fusion_iters(raito_idx, :)), mean(final_clustering_costs(raito_idx, :)), std(final_clustering_costs(raito_idx, :))] , '-append', 'delimiter', '\t', 'newline', 'pc');    
end

% save('protein_final_result.mat', 'final_clustering_accs', 'final_clustering_nmis', 'final_clustering_purities', 'final_clustering_fmeasures', 'final_clustering_ris', ...
%      'final_clustering_aris', 'final_clustering_iters', 'final_clustering_fusion_iters', 'final_clustering_costs', 'final_clustering_f_errors', 'final_clustering_s_errors', ... 
%     'individual_view_sparsity_ratios', 'sparsity_ratios_bf_fusion', 'sparsity_ratios_af_fusion');

