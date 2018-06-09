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


%% Naive Bayes

% 1. normal 1-dim in each coordinate, get mean and std
D = size(X,2);
malignant = labelTraining == 1;
non_malignant = ~malignant;
malignant_MLE = zeros(D, 2);
non_malignant_MLE = zeros(D, 2);
for i=1:D
    malignant_MLE(i,:) = normfit(dataTraining(malignant,i));
    non_malignant_MLE(i,:) = normfit(dataTraining(non_malignant,i));
end

% 2. estimate the apriori Class probability
P_malignant = sum(malignant) / length(labelTraining);
P_non_malignant = sum(non_malignant)/ length(labelTraining);


% 3.1 Naive Bayes estimate of the tags - Training
P_is = zeros(length(labelTraining),1);
P_isnt = zeros(length(labelTraining),1);
for k=1:length(labelTraining)
    P_x_given_is = 1; % needed for Pi notation
    P_x_given_non = 1; % needed for Pi notation
    for i=1:D
        P_x_given_is = P_x_given_is*normpdf(dataTraining(k,i), malignant_MLE(i,1), malignant_MLE(i,2));
        P_x_given_non = P_x_given_non*normpdf(dataTraining(k,i), non_malignant_MLE(i,1), non_malignant_MLE(i,2));
    end
        P_is(k) = P_x_given_is*P_malignant;
        P_isnt(k) = P_x_given_non*P_non_malignant;
end
training_tag = sign(P_is - P_isnt);
result_training = sum(training_tag==labelTraining) / length(labelTraining)

% 3.2 Naive Bayes estimate of the tags - Testing (same code different variables)
% for future use, can be refactored into a function
P_is = zeros(length(labelTesting),1);
P_isnt = zeros(length(labelTesting),1);
for k=1:length(labelTesting)
    P_x_given_is = 1; % needed for Pi notation
    P_x_given_non = 1; % needed for Pi notation
    for i=1:D
        P_x_given_is = P_x_given_is*normpdf(dataTesting(k,i), malignant_MLE(i,1), malignant_MLE(i,2));
        P_x_given_non = P_x_given_non*normpdf(dataTesting(k,i), non_malignant_MLE(i,1), non_malignant_MLE(i,2));
    end
        P_is(k) = P_x_given_is*P_malignant;
        P_isnt(k) = P_x_given_non*P_non_malignant;
end
Testing_tag = sign(P_is - P_isnt);
result_Testing = sum(Testing_tag==labelTesting) / length(labelTesting)
% Code Timing result = 
% Function Name, CallsTotal, Total Time,  Self Time
% hw_wet_bayes,      1,       0.116 s,     0.040 s
% Clasification Error for Train and Test
%result_training = 80.88% accuracy
%result_Testing = 84.21% accuracy