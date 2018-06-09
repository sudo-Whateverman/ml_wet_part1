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
for i=1:length(y)
    if y(i)==0
        y(i)= -1; % label as minus one for easier clasification
    end
end
dataTraining = X(tf,:);
labelTraining = y(tf,:);
dataTesting = X(~tf,:);
labelTesting = y(~tf,:);


%% Logistic Regression
