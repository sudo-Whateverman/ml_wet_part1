%% Wet excercise 1
% Made by Nick Kuhlik & Daniela Lipshitz


%% load data and split test and training

load('BreastCancerData.mat')
X = transpose(X);
N = size(X,1);  % total number of rows
k_batches = 10;

% testing Method No 1


% testing Method No 2
p = .8;    % proportion of rows to select for training
tf = false(N,1);    % create logical index vector
tf(1:round(p*N)) = true;
tf = tf(randperm(N));   % randomise order
dataTraining = X(tf,:);
labelTraining = y(tf,:);
dataTesting = X(~tf,:);
labelTesting = y(~tf,:);

%% K - means -

K = 2;
r = 10000000000;
epsilon = 1;
% [IDX, C, SUMD, D] = kmeans(X, K); - This is the needed result.
% first we may need to run the check i times, so we instantiate i slices
edges = round(linspace(1,K,N)); % Bin slices to indice of 1:K
for m=1:K
    edges_logical = (edges==m);
    data = X(~edges_logical, :);
    labels = y(~edges_logical, :);
    % 0. Take k central points, t =0
    % we randomly take a certain point because random works best
    centroid = datasample(data,K);
    a = 0;
    while(true)
        a = a + 1;
        % 1. Use the eucledean distance for each point to the centroid and tag it
        % accordingly
        distances_from_center = zeros(K, length(labels));
        for k = 1:K
            for i = 1:length(labels)
                distances_from_center(k, i) = my_euclidean_dist(data(i, :),centroid(k,:));
            end
        end
        
        tags = 1:length(labels);
        for i = 1:length(labels)
            [M,I] = min(distances_from_center(:, i));
            tags(i) = I;
        end
        
        
        % 2. Recalculate new centroid according to the central mass
        old_centroid = centroid;
        for i=1:K
            centroid(i,:) = mean(data(tags==i, :));
        end
        % 3. Repeat until distance(centroid(t), centroid(t-1)) < epsilon
        r = my_euclidean_dist(centroid,old_centroid);
        if (r < epsilon)
            break
        end
    end
    % check that Testing tags ?= Predicted
    Testing_data = X(edges_logical, :);
    Testing_labels = y(edges_logical, :);
    
    distances_from_center = zeros(K, length(Testing_labels));
    for k = 1:K
        for i = 1:length(Testing_labels)
            distances_from_center(k, i) = my_euclidean_dist(Testing_data(i, :),centroid(k,:));
        end
    end
    Testing_tags = 1:length(Testing_labels);
    for i = 1:length(Testing_labels)
        [M,I] = min(distances_from_center(:, i));
        Testing_tags(i) = I;
    end
    
    result = my_data_check(Testing_tags,transpose(Testing_labels))
    % Run i times each time taking n_th of i slice as testing
    
end




