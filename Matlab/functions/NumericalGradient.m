function [dW, db] = NumericalGradient(X, yOneHot, W, b, fList)
%% DESCRIPTION: Calculate numerical gradients for W and b for neural network
%---INPUT VARIABLE(S)---
%   (1) X: Matrix with explanatory variables (2xT)
%   (2) yOneHot: Matrix of labeled data in one-hot encoding (3xT)
%   (3) W: Cell array of weight matrices at which numerical gradient is calculated
%   (4) b: Cell array of bias vectors at which numerical gradient is calculated
%   (5) fList: Cell array of functions governing neural network
%   nonlinearities
%---OUTPUT VARIABLE(S)---
%   (1) dW: Cell array of numerical gradients for W (same size as W)
%   (2) db: Cell array of numerical gradients for b (same size as b)

    % Increment for numerical difference
    EPSILON = 1E-5;
    
    % Dimensions
    L = length(fList);

    % Initialize output
    for l = 1:L
        dW{l} = zeros( size(W{l}) );
        db{l} = zeros( size(b{l}) );
    end

    % Calculate gradients in b
    for l = 1:L
        nRows = length(b{l});
        for iter = 1:nRows

            % Set default values
            bEpsPlus = b; bEpsMinus = b;

            % Increment elements
            bEpsPlus{l}(iter) = bEpsPlus{l}(iter)+EPSILON;
            bEpsMinus{l}(iter) = bEpsMinus{l}(iter)-EPSILON;

            % Numerical gradient
            costPlus = Prop_Forward(X, yOneHot, W, bEpsPlus, fList);
            costMinus = Prop_Forward(X, yOneHot, W, bEpsMinus, fList);
            db{l}(iter) = (costPlus-costMinus)/(2*EPSILON);
        end
    end

    % Calculate gradients in W
    for l = 1:L
        [nRows, nCols] = size(W{l});
        for rowiter = 1:nRows
            for coliter = 1:nCols

                % Set default values
                WEpsPlus = W; WEpsMinus = W;

                % Increment elements
                WEpsPlus{l}(rowiter, coliter) = WEpsPlus{l}(rowiter, coliter)+EPSILON;
                WEpsMinus{l}(rowiter, coliter) = WEpsMinus{l}(rowiter, coliter)-EPSILON;

                % Nuermical gradient
                costPlus = Prop_Forward(X, yOneHot, WEpsPlus, b, fList);
                costMinus = Prop_Forward(X, yOneHot, WEpsMinus, b, fList);
                dW{l}(rowiter, coliter) = (costPlus-costMinus)/(2*EPSILON);
            end
        end
    end
end

