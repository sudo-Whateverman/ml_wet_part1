%% Wet excercise 1 kmeans
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

%% K - means seif 1

max_K = 12;
stopping = max_K;
error = 1:max_K;
error(1) = 10000000000;
tags_matrix = zeros(max_K,length(labels));

for K=2:max_K
    r = 10000000000;
    epsilon = 10;
    % [IDX, C, SUMD, D] = kmeans(X, K); - This is the needed result.
    data = X;
    labels = y;
    % 0. Take k central points, t =0
    % we randomly take a certain point because random works best
    centroid = datasample(data,K);
    
    
    iterations = 0;
    while(iterations < 1000)
        iterations = iterations+ 1; % failsafe to achieve convergence if epsilon is not achieved
        if ~iterations%100
            fprintf('K = %d and iter = %d \n',K, iterations)
        end
        % 1. Use the eucledean distance for each point to the centroid and tag it
        % accordingly
        distances_from_center = zeros(K, length(labels));
        for k = 1:K
            for i = 1:length(labels)
                distances_from_center(k, i) = my_eucledian_dist(data(i, :),centroid(k,:));
            end
        end
        
        tags = 1:length(labels);
        min_distances = 1:length(labels);
        for i = 1:length(labels)
            [M,I] = min(distances_from_center(:, i));
            min_distances(i) = M^2; % useful to calculate the error of the clustering
            tags(i) = I;
        end
        
        
        % 2. Recalculate new centroid according to the central mass
        old_centroid = centroid;
        for i=1:K
            centroid(i,:) = mean(data(tags==i, :));
        end
        
        % 3. Repeat until distance(centroid(t), centroid(t-1)) < epsilon
        r = my_eucledian_dist(centroid,old_centroid);
        if (r < epsilon)
            tags_matrix(K,:) = tags;
            break
        end
    end
    %% K - means seif 2

    % check the clustering error
    error(K)  = sqrt(sum(min_distances));
    if 1-error(K)/error(K-1) < 0.0001*epsilon
        fprintf('We have stopped at K= %d, error rate too low or revrsed\n',K)
        stopping = K;
        break
    end
    % The appropriate k is stopping -1 and it depends on the random choise
    % and the epsilon value for the error decline and the distance change
    % so in our case a good choise is between 5 and 9, while 7 is mostly a
    % knee of the error graph
end
figure
plot(2:stopping,error(2:stopping))
title('order of K clusters against clustering error')

%% K - means seif 3
[coeff, score] = pca(X);
x_axis = coeff(:,1);
y_axis = coeff(:,2);

% For K=2 we choose
figure()
tags_2_logical = tags_matrix(2,:)==1;
plot(score(tags_2_logical,1),score(tags_2_logical,2),'g+', score(~tags_2_logical,1),score(~tags_2_logical,2),'rx')
xlabel('1st Principal Component')
ylabel('2nd Principal Component')
title('pca components and data explained by K-means for k=2')

%% K - means seif 4

% For K=stopping-1 the one before the algo gives the ghost
figure()
for i=1:stopping-1
    tags_logical = tags_matrix(stopping-1,:)==i;
    plot(score(tags_logical,1),score(tags_logical,2),'color',rand(1,3),'marker', '+', 'LineStyle','none');
    hold on
end
xlabel('1st Principal Component')
ylabel('2nd Principal Component')
title(['pca components and data explained by K-means for k=',num2str(stopping-1)])
legend('show')
hold off

