function [ R ] = my_eucledian_dist(point,center)
% computes eucledean distance for n-dimensional vector
    R = 0;
    for i = 1:length(point)
        R = R + (point(i) - center(i))^2;
    end    
    R = sqrt(R);

end

%% Testing 
% my_eucledian_dist([1,0,0],[0, 0, 0])
% my_eucledian_dist([1,0],[0,0])