function [] = ConfusionMatrix(y, yhat, labellist)
%% DESCRIPTION: Initialize W and b for fully-connected neural network
%---INPUT VARIABLE(S)---
%   (1) y: Vector of length T containing true labels
%   (2) yhat: Vector of length T containing predicted labels
%   (3) labellist: Vector specifying the labels that are encountered in y
%   and yhat

% Check input for equal length
assert(length(yhat)==length(y), 'True labels and predicted labels are not of the same length.')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Print confusion matrix %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Header
fprintf('\n\n%-6s|', '')
for iter = 1:length(labellist)
    fprintf('%10s ', strcat('yhat=',num2str( labellist(iter) )));
end
fprintf('\n')
% Content
for iter1 = 1:length(labellist)
    fprintf('%-6s|', strcat('y=', num2str( labellist(iter1) )));
    for iter2 = 1:length(labellist)
        fprintf('%10d', sum( (y==labellist(iter1)).*(yhat==labellist(iter2))  ))
    end
    fprintf('\n')
end