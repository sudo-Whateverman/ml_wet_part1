%% Wet excercise 1 decision tree
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


%% decision tree

