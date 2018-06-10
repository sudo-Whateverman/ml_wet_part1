function [ R ] = kulbak_lieber_distance(p,q)
% computes eucledean kulbak_lieber_distance
    R = p*log(p/q);
end

%% Testing 
% kulbak_lieber_distance(1,1)
% kulbak_lieber_distance(1,0.5)