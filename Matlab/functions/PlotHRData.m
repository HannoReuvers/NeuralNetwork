function [] = PlotHRData(X, y)
%% DESCRIPTION: Plot HR initials data
%---INPUT VARIABLE(S)---
%   (1) X: Matrix with explanatory variables (2xT)
%   (2) y: Vector of length T specifying the label (1, 2, or 3) of each
%   observation

    % Request default colors
    ColorScheme = colororder;

    % Creat plot
    plot(X(1, y==3), X(2, y==3), 'o', 'MarkerSize', 5, 'MarkerFaceColor', [.7 .7 .7], 'MarkerEdgeColor', [.7 .7 .7])
    hold on
    plot(X(1, y==1), X(2, y==1), 'o', 'MarkerSize', 5, 'MarkerFaceColor', ColorScheme(1,:), 'MarkerEdgeColor', ColorScheme(1,:))
    plot(X(1, y==2), X(2, y==2), 'o', 'MarkerSize', 5, 'MarkerFaceColor', ColorScheme(2,:), 'MarkerEdgeColor', ColorScheme(2,:))
    xticks([0 0.5 1])
    yticks([0 0.5 1])
    hold off
    grid off
    box on
    set(gca, 'FontSize', 12)
    xlabel('$x_1$', 'Interpreter', 'latex','FontSize', 25) 
    ylabel('$x_2$', 'Interpreter', 'latex','FontSize', 25) 
end

