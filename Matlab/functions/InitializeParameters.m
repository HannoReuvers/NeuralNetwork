function [W, b] = InitializeParameters(UnitsInLayers, InitializeMethod)
%% DESCRIPTION: Initialize W and b for fully-connected neural network
%---INPUT VARIABLE(S)---
%   (1) UnitsInLayers: Vector with number of units in each layer. The
%   first element is the number of units in the input layer
%   (2) InitializeMethod: Specify method to randomly initialize elements
%   in W ('default'/'normalized')
%---OUTPUT VARIABLE(S)---
%   (1) W: Cell array with randomly initialized W matrices
%   (2) b: Cell array with zero initialized b vectors

    % Number of layer connections
    L = length(UnitsInLayers)-1;

    for l = 1:L

        % W: random initialization
        if strcmp(InitializeMethod, 'default')
            % See (1) in Glorot and Bengio (2010)
            UnifBound = sqrt( 1/UnitsInLayers(l) ); 
        elseif strcmp(InitializeMethod, 'normalized')
            % See (16) in Glorot and Bengio (2010)
            UnifBound = sqrt( 6/(UnitsInLayers(l)+UnitsInLayers(l+1)) );
        else
            error('Invalid method to initialize W and b')
        end
        W{l} = -UnifBound + (2*UnifBound)*rand(UnitsInLayers(l+1), UnitsInLayers(l));

        % b: zero initialization
        b{l} = zeros(UnitsInLayers(l+1), 1);

    end
end

