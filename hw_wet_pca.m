load('BreastCancerData.mat')
count=0;
for i= 1:length(y)
    if y(i,:) == 1
        count=count+1;
    else
        y(i,:) = -1;
    end
end
mapcaplot(transpose(X),y)