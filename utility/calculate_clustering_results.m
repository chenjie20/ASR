function [acc, nmi, purity, fmeasure, ri, ari] = calculate_clustering_results(Z, groundLables, numClusters)

        groundLables = groundLables(:);
        currentLabels = new_spectral_clustering(Z, numClusters);     
        currentLabels = currentLabels(:);
        
%         acc = compute_accuracy(groundLables, currentLabels, numClusters);
         
        acc = accuracy(groundLables, currentLabels);
        
%         [sortedLabels] = bestMap(groundLables, currentLabels);
%         acc = mean(groundLables==sortedLabels);
                
        class_labels = zeros(1, numClusters);
        for idx =  1 : numClusters
            class_labels(idx) = length(find(groundLables == idx));
        end
        cluster_data = cell(1, numClusters);
        for idx =  1 : numClusters
            cluster_data(1, idx) = { groundLables(currentLabels == idx)' };
        end
        [nmi, purity, fmeasure, ri, ari] = calculate_results(class_labels, cluster_data);

end
