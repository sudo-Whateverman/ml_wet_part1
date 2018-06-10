function [ R ] = kulbak_lieber_distance(p,q)
% computes eucledean kulbak_lieber_distance
    R = p*log(p/q);
end