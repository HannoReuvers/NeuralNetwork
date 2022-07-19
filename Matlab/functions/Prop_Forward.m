function [cost, a, z] = Prop_Forward(X, yOneHot, W, b, fList)
%% DESCRIPTION: Forward propagation of neural network
%---INPUT VARIABLE(S)---
%   (1) X: Matrix with explanatory variables (2xT)
%   (2) yOneHot: Matrix of labeled data in one-hot encoding (3xT)
%   (3) W: Cell array of weight matrices at which numerical gradient is calculated
%   (4) b: Cell array of bias vectors at which numerical gradient is calculated
%   (5) fList: Cell array of functions governing neural network
%   nonlinearities
%---OUTPUT VARIABLE(S)---
%   (1) cost: Value of cross-entropy loss function
%   (2) a: Cell array of all activations in neural network
%   (3) z: Cell array of all linear combinations in neural network

    % Dimensions
    L = length(fList);
    n = size(yOneHot, 2);

    % Initialize a and z
    for l = 1:L
        a{l} = zeros( size(b{l}) );
        z{l} = zeros( size(b{l}) );
    end

    % Forward propagation
    for l=1:L
        % Note: implicit expansion will add bias to each column

        % Layer 1 receiving data input
        if l==1
            z{l} = W{l}*X+b{l};
            a{l} = fList{l}( z{l} );
        else
            z{l} = W{l}*a{l-1}+b{l};
            a{l} = fList{l}( z{l} );
        end
    end
    
    % Cost function
    cost = - sum( yOneHot.*log(a{L}), 'all' )/n;
end

