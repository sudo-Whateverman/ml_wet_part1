%% Wet excercise 1 bayes
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


%% Logistic Regression - batch method - 3 different learning speeds

% 0.0 Init stopping rules
max_iter = length(labelTraining)*100;
% 0.1 random weights for C1 and C2
w = zeros(1,size(X,2)+1);
% 0.2 random permutation of Training
p = randperm(length(labelTraining));
randomLabelsTraining = labelTraining(p);
% 0.2.1 NORMALIZE DATA:
for i=1:size(X,2)
    dataTraining(:,i) = 2*(dataTraining(:,i) - min(dataTraining(:,i))) / ( max(dataTraining(:,i)) - min(dataTraining(:,i)) ) - 1;
end
randomDataTraining = [ones(length(labelTraining),1), dataTraining(p,:)]; % add ones to the first column
% 0.3 init thresh and slope param
thresh = 0.0005;
slope = [ 0.3, 0.1, 0.05] * 0.01;

clear error_graph;
clear error_graph_Testing;
clear thresh_graph;
clear stoptime;
% WHILE NOT STOP FOR EACH VALUE:
for k =1:length(slope)
    pass = false;
    cur_iter = 1;
    small_iter = 1;
    flag = false;
    tags = zeros(length(labelTraining),1);
    phi = zeros(length(labelTraining),1);
    phi_tag = zeros(length(labelTraining),1);
    while(true)
        for index=1:length(labelTraining)
            small_iter = small_iter + 1;
            % 1.0 predict tag
            v = w*transpose(randomDataTraining(index,:));
            phi(index) = 1 / (1 + exp(-v));
            if phi(index) >= 0.5
                tags(index) = 1;
            else
                tags(index) = 0;
            end
            % 1.1 update weights - gradient descent
            phi_tag(index) = exp(-v) / (1 + exp(-v))^2;
        end
        w_old = w;
        guess = tags - randomLabelsTraining;
        delta_w = 0;
        for index=1:length(labelTraining)
            delta_w = delta_w + guess(index)*phi_tag(index)*randomDataTraining(index,:);
        end
        delta_w = -slope(k)*delta_w;
        size_of_delta = my_eucledian_dist(delta_w,zeros( 1, size(X,2)+1));
        if  size_of_delta ~= 0
            w = w + delta_w;
            
            % 1.2 check thresh and error convergence (if at least 1 pass - break)
            thresh_val = my_eucledian_dist(w_old,w);
            error = sum(tags~=randomLabelsTraining);
            % 1.2.1 get error convergence for each iteration for train
            error_graph(k,cur_iter) = error;
            thresh_graph(k,cur_iter) = thresh_val;
            if thresh_val < thresh
                flag = true;
                stoptime(k) = cur_iter;
            end
            if cur_iter > max_iter || small_iter > max_iter * 10
                flag = true;
                stoptime(k) = cur_iter;
            end
            % check the testing accuracy for W
            error_test = check_testing(w,dataTesting,labelTesting);
            error_graph_Testing(k,cur_iter) = error_test;
            cur_iter = cur_iter+1;
            
        end
        if flag == true
            %                 fprintf('we are exiting the loop\n')
            break
        end
        
    end
    pass = true;

    
    % try new W
    w= zeros(1,size(X,2)+1);
    p = randperm(length(labelTraining));
    randomLabelsTraining = randomLabelsTraining(p);
    randomDataTraining = randomDataTraining(p,:);
    
end
figure()
hold on
plot(1:stoptime(3), error_graph(3,1:stoptime(3)),'b')
plot(1:stoptime(2), error_graph(2,1:stoptime(2)),'g')
plot(1:stoptime(1), error_graph(1,1:stoptime(1)),'r')

legend('slope=0.5','slope=1','slope=4') % reverse order
title('Training error vs iteration for different slopes')
hold off

figure()
hold on
plot(1:stoptime(3), error_graph_Testing(3,1:stoptime(3)),'b')
plot(1:stoptime(2), error_graph_Testing(2,1:stoptime(2)),'g')
plot(1:stoptime(1), error_graph_Testing(1,1:stoptime(1)),'r')

legend('slope=0.5','slope=1','slope=4') % reverse order
title('Testing error vs iteration for different slopes')
hold off

% AFTER
% 2.0 print results for training and testing
result_training = 1 - error_graph(1,stoptime(1))/length(randomLabelsTraining);
fprintf(['Training result for slope ',num2str(slope(1)),' is ',num2str(result_training*100),'\n'])
result_training = 1 - error_graph(2,stoptime(2))/length(randomLabelsTraining);
fprintf(['Training result for slope ',num2str(slope(2)),' is ',num2str(result_training*100),'\n'])
result_training = 1 - error_graph(3,stoptime(3))/length(randomLabelsTraining);
fprintf(['Training result for slope ',num2str(slope(3)),' is ',num2str(result_training*100),'\n'])
result_training = 1 - error_graph_Testing(1,stoptime(1))/length(randomLabelsTraining);
fprintf(['Testing result for slope ',num2str(slope(1)),' is ',num2str(result_training*100),'\n'])
result_training = 1 - error_graph_Testing(2,stoptime(2))/length(randomLabelsTraining);
fprintf(['Testing result for slope ',num2str(slope(2)),' is ',num2str(result_training*100),'\n'])
result_training = 1 - error_graph_Testing(3,stoptime(3))/length(randomLabelsTraining);
fprintf(['Testing result for slope ',num2str(slope(3)),' is ',num2str(result_training*100),'\n'])
