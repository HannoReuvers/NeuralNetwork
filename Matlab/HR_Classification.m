 clc ; clear variables; close all;
addpath("./functions")
rng(1234)

% Read data
[Ttrain, Xtrain, ytrain, yOneHottrain] = ReadHRInitialsData('train');
[Tvalid, Xvalid, yvalid, yOneHotvalid] = ReadHRInitialsData('valid');
[Ttest, Xtest, ytest, yOneHottest] = ReadHRInitialsData('test');

% Activation functions
sigmoid = @(x) 1./ (1+exp(-x));
softmax = @(x) exp(x) ./ sum( exp(x), 1);
Tanh = @(x) tanh(x);
ReLU = @(x) x.*(x>0);

% Gradient functions
gradsigmoid = @(x) sigmoid(x).*(1-sigmoid(x));
gradTanh = @(x) 1-Tanh(x).^2;
gradReLU = @(x) 1.0*(x>0);

% Network architecture
n0 = size(Xtrain, 1);
n1 = 50;
n2 = 50;
n3 = 3;
Units = [n0, n1, n2, n3];
L = length(Units)-1;
FunctionList = {Tanh, Tanh, softmax};
GradList = {gradTanh, gradTanh};

% Neural network hyperparameters
LearningRate = 0.2;
NumberOfEpochs = 2000;
MaxEpochs = 20000;          % Maximum number of iterations

% Initialize parameters and network
[W, b] = InitializeParameters(Units, 'normalized');
for l=1:L
    a{l} = zeros(Units(l+1), Ttrain);
    z{l} = zeros(Units(l+1), Ttrain);
end

%%%------- GRADIENT CHECK (OPTIONAL) -------%%% 
GradCheck = 1;
if GradCheck==1
    % Create smaller network
    TempUnits = [n0, 10, 5, n3];
    % Initialize smaller network
    [Wtest, btest] = InitializeParameters(TempUnits, 'normalized');
    % Numerical gradient
    [NumericaldW, Numericaldb] = NumericalGradient(Xtrain, yOneHottrain, Wtest, btest, FunctionList);
    [~, a, z] = Prop_Forward(Xtrain, yOneHottrain, Wtest, btest, FunctionList);
    [dW, db, ~] = Prop_Backward(Xtrain, yOneHottrain, Wtest, a, z, GradList);
    for l=1:L
        fprintf('Layer %d gradient difference norms: %5.3f %5.3f\n', l, norm(NumericaldW{l}-dW{l}), norm(Numericaldb{l}-db{l}))
    end
    fprintf('\n')
end


% Initialize lists
CostListTrain = NaN(MaxEpochs, 1);
CostListValid = NaN(MaxEpochs, 1);

for epoch = 1:MaxEpochs

    % Forward propagation
    [costTrain, a, z] = Prop_Forward(Xtrain, yOneHottrain, W, b, FunctionList);
    [costValid, ~, ~] = Prop_Forward(Xvalid, yOneHotvalid, W, b, FunctionList);
    % Inform user
    fprintf('Training cost in epoch %d: %f\n', epoch, costTrain)
    % Save results
    CostListTrain(epoch) = costTrain;
    CostListValid(epoch) = costValid;

    % Backward propagation
    [dW, db, da] = Prop_Backward(Xtrain, yOneHottrain, W, a, z, GradList);

    % Gradient descent step
    for l = 1:L
        b{l} = b{l} - LearningRate*db{l};
        W{l} = W{l} - LearningRate*dW{l};
    end

    % Show intermediate performance
    if epoch==NumberOfEpochs

        % Plot 1: Cost function for training data
        figure(1)
        plot(CostListTrain(1:NumberOfEpochs), 'o');
        xticks([0 2000 4000 6000 8000])
        yticks([0 0.2 0.4 0.6 0.8 1.0 1.2])
        grid on
        box on
        set(gca, 'FontSize', 12)
        xlabel('learning epoch', 'FontSize', 25)
        ylabel('training cost', 'FontSize', 25) 
        drawnow;
        % Plot 2: Cost function for validation data
        figure(2)
        plot(CostListValid(1:NumberOfEpochs), 'o');
        xticks([0 2000 4000 6000 8000])
        yticks([0 0.2 0.4 0.6 0.8 1.0 1.2])
        grid on
        box on
        set(gca, 'FontSize', 12)
        xlabel('learning epoch', 'FontSize', 25)
        ylabel('validation cost', 'FontSize', 25) 
        drawnow;
        % Plot 3: Current decisions
        figure(3)
        DrawDecisionBoundary(yOneHottrain, W, b, FunctionList)

        % More iterations needed?
        fprintf('\n')
        EpochIncrement = input("Please increase the number of iterations by... (enter 0 to stop): ");
        fprintf('\n')
        NumberOfEpochs = NumberOfEpochs+EpochIncrement;

        % Close figures
        close(1); close(2); close(3);
    end

    % Check for completion
    if epoch==NumberOfEpochs
        break
    end
end

% Test set confusion matrix
[~, aTest, ~] = Prop_Forward(Xtest, yOneHottest, W, b, FunctionList);
[~, yhattest] = max(aTest{3});
ConfusionMatrix(ytest, yhattest, [1; 2; 3])


