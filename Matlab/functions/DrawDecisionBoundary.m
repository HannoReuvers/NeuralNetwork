function [] = DrawDecisionBoundary(yOneHot, W, b, fList)
%% DESCRIPTION: Draw decision boundary for HR initials data
%---INPUT VARIABLE(S)---
%   (1) yOneHot: Matrix of labeled data in one-hot encoding (3xT)
%   (2) W: Cell array of current weight matrices
%   (3) b: Cell array of current bias vectors
%   (4) fList: Cell array of functions governing neural network
%   nonlinearities

    % Predictions on grid
    x1min = 0; x1max = 1; x2min = 0; x2max = 1;
    incr = 0.01;
    x1Range = x1min:incr:x1max;
    x2Range = x2min:incr:x2max;
    [xx1, xx2] = meshgrid(x1Range,x2Range);
    XGrid = [xx1(:) xx2(:)];
    PredValues = NaN(size(XGrid,1), 1);
    for point = 1:size(XGrid, 1)
        coordinates = XGrid(point, :)';
        [~, Activation, ~] = Prop_Forward(coordinates, yOneHot, W, b, fList);
        % Assign label based on majority vote
        [~, PredLabel] = max(Activation{3});
        PredValues(point) = PredLabel;
    end
    
    ColorScheme = colororder;
    hold on
    % Create canvas
    scatter(XGrid(PredValues==1, 1), XGrid(PredValues==1, 2), 'o', 'MarkerEdgeColor', ColorScheme(1,:), 'MarkerFaceColor', ColorScheme(1,:), 'MarkerEdgeAlpha', 0.1, 'MarkerFaceAlpha', 0.1)
    scatter(XGrid(PredValues==2, 1), XGrid(PredValues==2, 2), 'o', 'MarkerEdgeColor', ColorScheme(2,:), 'MarkerFaceColor', ColorScheme(2,:), 'MarkerEdgeAlpha', 0.1, 'MarkerFaceAlpha', 0.1)
    scatter(XGrid(PredValues==3, 1), XGrid(PredValues==3, 2), 'o', 'MarkerEdgeColor', [.7 .7 .7], 'MarkerFaceColor', [.7 .7 .7], 'MarkerEdgeAlpha', 0.1, 'MarkerFaceAlpha', 0.1)
    xticks([0 0.5 1])
    yticks([0 0.5 1])
    hold off
    grid off
    box on
    set(gca, 'FontSize', 12)
    xlabel('$x_1$', 'Interpreter', 'latex','FontSize', 25)
    ylabel('$x_2$', 'Interpreter', 'latex','FontSize', 25) 
    drawnow;
end

