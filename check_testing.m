function [ error ] = check_testing(w,dataTesting,labelTesting)
% computes error of logistic regression
predict = labelTesting;
% 0.2.1 NORMALIZE DATA:
for i=1:size(dataTesting,2)
    dataTesting(:,i) = 2*(dataTesting(:,i) - min(dataTesting(:,i))) / ( max(dataTesting(:,i)) - min(dataTesting(:,i)) ) - 1;
end
dataTesting = [ones(length(labelTesting),1), dataTesting(:,:)]; % add ones to the first column
for index=1:length(labelTesting)
    
    v = w*transpose(dataTesting(index,:));
    phi = 1 / (1 + exp(-v));
    
    if phi >= 0.5
        predict(index) = 1;
    else
        predict(index) = 0;
    end
end
error = sum(labelTesting~=predict);
end

%% Testing
% kulbak_lieber_distance(1,1)
% kulbak_lieber_distance(1,0.5)