
function R = my_eucledian_dist(point,center)
    R = 0;
    for i = 1:length(point)
        R = R + (point(i) - center(i))^2;
    end    
    R = sqrt(R);
end

%% Testing 
% my_euclidean_dist([1,0,0],[0, 0, 0])
% my_euclidean_dist([1,0],[0,0])