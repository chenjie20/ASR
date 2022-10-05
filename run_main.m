close all;
clear;
clc;

addpath('data');
addpath('utility');

%--We may a PCA algorithm to preprocess the original features of all samples
% a PCA algorithm will be skippend when the parameter is 0 ---------------
new_dims = [0, 0, 0, 0];

data_index = 1;
switch data_index
    case 1
        filename = "BBC";
        load('BBC4view_685.mat');
        n = size(truelabel{1}, 2);
        nv = size(data, 2);
        K = length(unique(truelabel{1}));
        gnd = truelabel{1};
        data_views = cell(1, nv);
        for nv_idx = 1 : nv
             data_views{nv_idx} = data{nv_idx};
        end
        new_dims = [600, 600, 400, 300];          
        asr_parameters = [1e-6,20;1e-6,20;1e-6,20;1e-6,20];
        sparse_fusion_used = true;

    case 2
        filename = "reuters";
        load('reuters.mat');
        K = size(category, 2);
        nv = size(X, 2);
        data_views = cell(1, nv);
        num_each_class = 100;
        total_num = num_each_class * K;
        n = total_num;
        gnd = zeros(1, total_num);
        for nv_idx = 1 : nv 
             dim = size(X{nv_idx}, 2);
             data_views{nv_idx} = zeros(dim, num_each_class * K);
        end
        for idx = 1 : K
           view_ids = find(Y == (idx - 1));
           len = length(view_ids);
           rand('state', 2000);
           rnd_idx = randperm(len);
           new_view_ids = view_ids(rnd_idx(1 :num_each_class));
           current_ids = ((idx - 1) * num_each_class + 1) : idx * num_each_class;
           gnd(1, current_ids) = idx;
           for nv_idx = 1 : nv 
               data_views{nv_idx}(:, current_ids) = X{nv_idx}(new_view_ids, :)';
           end
        end
        new_dims = [0, 500, 400, 200];
        asr_parameters = [5e-6,5;5e-6,5;5e-6,1e-4;1e-4,20];
        sparse_fusion_used = false;

    case 3
        filename = "flower17";
        load('flower17_Kmatrix.mat');
        n = length(Y);
        nv = size(KH, 3);
        K = length(unique(Y));
        gnd = Y;
        data_views = cell(1, nv);
        for nv_idx = 1 : nv
             data_views{nv_idx} = KH(:, :, nv_idx);
        end
        new_dims = [200, 100, 100, 100];  
        asr_parameters = [1e-7,1e-4;1e-7,1e-4;1e-7,1e-4;1e-4,1e-4];
        sparse_fusion_used = true;

    case 4
        filename = "proteinFold";
        load('proteinFold_Kmatrix.mat');
        n = length(Y);
        nv = size(KH, 3);
        K = length(unique(Y));
        gnd = Y;
        data_views = cell(1, nv);
        for nv_idx = 1 : nv
             data_views{nv_idx} = KH(:, :, nv_idx);
        end
        new_dims = [50, 50, 50, 50];
        asr_parameters = [1e-6,0.07;1e-6,0.07;1e-5,0.05;1e-5,0.05];
        sparse_fusion_used = true;

   case 5
        filename = "leaves";
        load('100leaves.mat');
        n = size(truelabel{1}, 1);
        nv = size(data, 2);
        K = length(unique(truelabel{1}));
        gnd = truelabel{1};
        data_views = cell(1, nv);
        for nv_idx = 1 : nv            
            data_views{nv_idx} = data{nv_idx};
        end
        asr_parameters = [1e-7,20;1e-7,10;0.1,300;0.1,300];
        sparse_fusion_used = false;

    case 6
        filename = "Caltech101";
        load('Caltech101.mat');
        nv = size(fea, 2);
        gnd = gt';
        
        %We removed the background category.
        positions = find(gnd > 1);
        gnd = gnd(positions);
        K = length(unique(gnd));
        gnd = gnd - 1;
        
        caltech_data_views = cell(1, nv);        
        for nv_idx = 1 : nv
             tmp = fea{nv_idx}';
             caltech_data_views{nv_idx} = tmp(:, positions);
        end
        n = length(gnd);        
        new_dims = [200, 200, 200, 200];
        asr_parameters = [1e-8,0.1;1e-8,0.1;1e-6,5e-3;1e-4,0.1];
        sparse_fusion_used = true;

    case 7
        filename = "MSRCv1";
        load('MSRCv1.mat');
        n = length(Y);
        nv = size(X, 2);
        K = length(unique(Y));
        gnd = Y;
        data_views = cell(1, nv);
        for nv_idx = 1 : nv      
            data_views{nv_idx} = X{nv_idx}';           
        end
        data_views = normalize_multiview_data(data_views);
        asr_parameters = [0.005,1;0.01,5;0.01,5;0.005,2];
        sparse_fusion_used = true;
  
    case 8
        filename = "COIL20";
        load('COIL20.mat');
        n = length(Y);
        nv = size(X, 2);
        K = length(unique(Y));
        gnd = Y;
        data_views = cell(1, nv);
        for nv_idx = 1 : nv
            data_views{nv_idx} = X{nv_idx}';            
        end
        data_views = normalize_multiview_data(data_views);
        asr_parameters = [0.1,1000;0.01,0.1;0.0005,0.1;0.01,0.5];
        sparse_fusion_used = false;

    case 9
        filename = "handwritten";
        load('handwritten.mat');
        n = length(Y);
        nv = size(X, 2);
        K = length(unique(Y));
        gnd = Y + 1;
        data_views = cell(1, nv);
        for nv_idx = 1 : nv
             data_views{nv_idx} = X{nv_idx}';
        end
        asr_parameters = [1e-05,0.05;1e-05,0.1;0.0001,100;0.0005,20];
        sparse_fusion_used = true;

    case 10
        filename = "Scene15";
        load('Scene15.mat');
        n = length(Y);
        nv = size(X, 2);
        K = length(unique(Y));
        gnd = Y;
        data_views = cell(1, nv);
        for nv_idx = 1 : nv
             data_views{nv_idx} = X{nv_idx}';
        end
        asr_parameters = [1e-6,10;1e-6,10;5e-06,50;5e-06,5];
        sparse_fusion_used = false;
end

%---------- The missing ratios --------------------------------------------
%- There are four experiments. --------------------------------------------
%--For example-------------------------------------------------------------
%- 0 represents that all features are available.
%- 0.1 represents that 10% of features are randomly missing in each view.
%-------------------------------------------------------------------------
missing_raitos = [0, 0.1, 0.3, 0.5];

class_labels = zeros(1, K);
for idx =  1 : K
    class_labels(idx) = length(find(gnd == idx));
end

% final_result = strcat(filename, '_par_result_ASR.txt');
% final_average_result = strcat(filename, '_par_avg_result_ASR.txt');

final_result = strcat(filename, '_result_ASR.txt');
final_average_result = strcat(filename, '_avg_result_ASR.txt');

%----------- The following parameters remian unchanged ---------------------
% sparsity_ratio = 0.2;
iter2 = 0;
max_iters = 500;
errors2 = zeros(1, max_iters);
ratio_len = length(missing_raitos);

% We can set the varaiable "repeated_times" to 10 for choosing 
%   the best result in each dataset.
repeated_times = 10; % finding the proper parameters quickly by setting repeated_times to 1
if repeated_times > 1
    final_clustering_accs = zeros(ratio_len, repeated_times);
    final_clustering_nmis = zeros(ratio_len, repeated_times);
    final_clustering_purities = zeros(ratio_len, repeated_times);
    final_clustering_fmeasures = zeros(ratio_len, repeated_times);
    final_clustering_ris = zeros(ratio_len, repeated_times);
    final_clustering_aris = zeros(ratio_len, repeated_times);
    final_clustering_costs = zeros(ratio_len, repeated_times);
    final_clustering_iters = zeros(ratio_len, repeated_times);
    final_clustering_values = zeros(ratio_len,  repeated_times);
end

Mn = cell(1, nv);
for raito_idx = 1 : length(missing_raitos)
    stream = RandStream.getGlobalStream;
    reset(stream);
    missing_raito = missing_raitos(raito_idx);
    raito = 1 - missing_raito;    
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

    alpha = asr_parameters(raito_idx, 1);
    beta = asr_parameters(raito_idx, 2);

    tic;
    [Wn, iter1, errors1] = asr(data_views, Mn, alpha, beta, new_dims(raito_idx));
    if sparse_fusion_used
        [Zs, ~, ~] = graph_fusion(Wn);
    else
        Zs = simple_graph_fusion(Wn);
    end
    num_nonzeros = sum(sum(abs(Zs) > 1e-6));
    current_sparsity_ratio = num_nonzeros / (n * n); 
    W = (abs(Zs) + abs(Zs')) / 2;
    disp([raito_idx, current_sparsity_ratio]);
    time_cost = toc;
    for time_idx = 1 : repeated_times
        [acc, nmi, purity, fmeasure, ri, ari] = calculate_clustering_results(W, gnd, K);        
        final_clustering_accs(raito_idx, time_idx) = acc;
        final_clustering_nmis(raito_idx, time_idx) = nmi;
        final_clustering_purities(raito_idx, time_idx) = purity;
        final_clustering_fmeasures(raito_idx, time_idx) = fmeasure;
        final_clustering_ris(raito_idx, time_idx) = ri;
        final_clustering_aris(raito_idx, time_idx) = ari;
        final_clustering_iters(raito_idx, time_idx) = iter1;
        final_clustering_costs(raito_idx, time_idx) = time_cost;
        disp([missing_raito, time_idx, alpha, beta, acc, nmi, purity, fmeasure, ri, ari]);
        writematrix([missing_raito, alpha, beta, roundn(acc, -2), roundn(nmi, -4), roundn(purity, -4), roundn(fmeasure, -4), roundn(ri, -4), roundn(ari, -4), roundn(time_cost, -2), iter1], final_result, "Delimiter", 'tab', 'WriteMode', 'append');       
    end

    if repeated_times > 1        
        averge_acc = mean(final_clustering_accs(raito_idx, :));
        std_acc = std(final_clustering_accs(raito_idx, :));
        averge_nmi = mean(final_clustering_nmis(raito_idx, :)); 
        std_nmi =std(final_clustering_nmis(raito_idx, :));
        averge_purity = mean(final_clustering_purities(raito_idx, :));
        std_purity =std(final_clustering_purities(raito_idx, :));
        averge_fmeasure = mean(final_clustering_fmeasures(raito_idx, :));
        std_fmeasure =std(final_clustering_fmeasures(raito_idx, :));
        averge_ri =  mean(final_clustering_ris(raito_idx, :));
        std_ri =std(final_clustering_ris(raito_idx, :));
        averge_ari = mean(final_clustering_aris(raito_idx, :));
        std_ari =std(final_clustering_aris(raito_idx, :));
        averge_cost = mean(final_clustering_costs(raito_idx, :)); 
        averge_iter = mean(final_clustering_iters(raito_idx, :));
    
        writematrix([missing_raito, alpha, beta, roundn(averge_acc, -2), roundn(std_acc, -2), roundn(averge_nmi, -4), roundn(std_nmi, -4), ...
            roundn(averge_purity, -4), roundn(std_purity, -4), roundn(averge_fmeasure, -4), roundn(std_fmeasure, -4), roundn(averge_ri, -4), roundn(std_ri, -4),...
            roundn(averge_ari, -4), roundn(std_ari, -4), roundn(averge_cost, -2), roundn(averge_iter, -2)], final_average_result, "Delimiter", 'tab', 'WriteMode', 'append'); 
        disp([missing_raito, alpha, beta, averge_acc, averge_nmi, averge_fmeasure]);
    end
end
