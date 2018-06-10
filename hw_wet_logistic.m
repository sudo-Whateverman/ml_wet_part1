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


%% Logistic Regression - online method - 3 different learning speeds

% 0.0 Init stopping rules
max_iter = length(labelTraining)*100;
% 0.1 random weights for C1 and C2
pd = makedist('Uniform');
w = random(pd, 1, size(X,2)+1);
% 0.2 random permutation of Training
p = randperm(length(labelTraining));
randomLabelsTraining = labelTraining(p);
% 0.2.1 NORMALIZE DATA:
for i=1:size(X,2)
    dataTraining(:,i) = (dataTraining(:,i) - min(dataTraining(:,i))) / ( max(dataTraining(:,i)) - min(dataTraining(:,i)) );
end
randomDataTraining = [ones(length(labelTraining),1), dataTraining(p,:)]; % add ones to the first column
% 0.3 init thresh and slope param
%thresh = [ 100, 1, 0.01] * 0.01;
thresh = 1;
slope = 0.001;
% WHILE NOT STOP FOR EACH VALUE:
for k =1:length(thresh)
    pass = false;
    cur_iter = 1;
    small_iter = 1;
    flag = false;
    tags = zeros(length(labelTraining),1);
    error_rate = 0;
    while(cur_iter < max_iter && small_iter < max_iter * 10)
        for index=1:length(labelTraining)
            small_iter = small_iter + 1;
            % 1.0 predict tag
            v = w*transpose(randomDataTraining(index,:));
            phi = 1 / (1 + exp(-v));
            if phi >= 0.5
                tags(index) = 1;
            else
                tags(index) = 0;
            end
            % 1.1 update weights - gradient descent
            phi_tag = exp(-v) / (1 + exp(-v))^2;
            w_old = w;
            guess = randomLabelsTraining(index) - phi;
            delta_w = (-slope*(guess)*phi_tag)*randomDataTraining(index,:);
            size_of_delta = my_eucledian_dist(delta_w,zeros( 1, size(X,2)+1));
            if  size_of_delta ~= 0
                w = w + delta_w;
                cur_iter = cur_iter+1;
                
                % 1.2 check thresh and error convergence (if at least 1 pass - break)
                thresh_val = my_eucledian_dist(w_old,w);
                thresh_graph(k,cur_iter) = thresh_val;
                if thresh_val < thresh(k)
                    if pass == true
                        flag = true;
                        stoptime(k) = cur_iter;
                    end
                end
            end
%             % 1.2.1 get error convergence for each iteration for test and train
%             error_rate(cur_iter) = error_rate(cur_iter-1) + tags(index) == randomLabelsTraining(index);
            % TODO
            if flag == true
                fprintf('we are exiting the loop\n')
                break
            end
        end
        pass = true;
        if flag == true
            continue
        end
    end
    % try new W
%     w = random(pd, 1, size(X,2)+1);
%     p = randperm(length(labelTraining));
%     randomLabelsTraining = randomLabelsTraining(p);
%     randomDataTraining = randomDataTraining(p,:);
    
end
figure()
hold on
plot(1:length(thresh_graph(1,:)), thresh_graph(1,:))
%plot(1:length(thresh_graph(2,:)), thresh_graph(2,:))
%plot(1:length(thresh_graph(3,:)), thresh_graph(3,:))
%legend('threshold=1', 'threshold=0.01', 'threshold=10e-4')
hold off
% AFTER
% 2.0 check results for training and testing

%% Logistic Regression - batch method

% 0.0 Init stopping rules

% 0.1 random weights for C1 and C2

% 0.2 random permutation of Training

% 0.3 init thresh and slope param

% WHILE NOT STOP FOR PASS on all values!:
% 1.0 predict tag

% 1.1 update weights

% 1.2 check thresh and error convergence (if at least 1 pass)

% 1.3 get different random permutation of data
p = randperm(length(labelTraining));
randomLabelsTraining = randomLabelsTraining(p);
randomDataTraining = randomDataTraining(p,:);
% AFTER
% 2.0 check results for training and testing

% TODO : add runtime