function [dW, db, da] = Prop_Backward(X, yOneHot, W, a, z, gradList)
%% DESCRIPTION: Backward propagation of neural network
%---INPUT VARIABLE(S)---
%   (1) X: Matrix with explanatory variables (2xT)
%   (2) yOneHot: Matrix of labeled data in one-hot encoding (3xT)
%   (3) W: Cell array of weight matrices at which numerical gradient is calculated
%   (4) a: Cell array of all activations in neural network
%   (5) z: Cell array of all linear combinations in neural network
%   (5) gradList: Cell array of gradient functions governing neural network
%   nonlinearities
%---OUTPUT VARIABLE(S)---
%   (1) dW: Cell array with gradients for weight matrices
%   (2) db: Cell array with gradients for bias vectors
%   (3) da: Cell array with gradients for activations

    % Dimensions
    L = length(W);
    n = size(yOneHot, 2);

    % Initialize dW, db, da, and dz
    for l = 1:L
        dW{l} = zeros( size(W{l}) );
        db{l} = zeros( size(a{l}) );
        da{l} = zeros( size(a{l}, n) );
        dz{l} = zeros( size(a{l}, n) );
    end

    % Softmax layer L
    dz{L} = a{L}-yOneHot;
    db{L} = sum(dz{L}, 2)/n;
    dW{L} = ( dz{L}*(a{L-1}') )/n;
    
    % Fully connected layers 1,...,L-1
    for l = (L-1):-1:1
        da{l} = W{l+1}'*dz{l+1};
        dz{l} = da{l}.*gradList{l}( z{l} );
        db{l} = sum(dz{l}, 2)/n;
        if l>1
            dW{l} = ( dz{l}*(a{l-1}'))/n;
        elseif l==1
            dW{l} = ( dz{l}*(X'))/n;
        end
    end
end

