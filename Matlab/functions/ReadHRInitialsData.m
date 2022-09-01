function [T, X, y, yOneHot] = ReadHRInitialsData(DataType)
%% DESCRIPTION: Read data from SQL file
%---INPUT VARIABLE(S)---
%   (1) DataType: Read 'train', 'valid', or 'test' data
%---OUTPUT VARIABLE(S)---
%   (1) T: Sample size
%   (2) X: Matrix with explanatory variables (2xT)
%   (3) y: Vector of length T with label 1, 2, or 3.
%   (4) yOneHot: Matrix of labeled data in one-hot encoding (3xT)

    % Determine SQL file to read
    if strcmp(DataType, 'train')
        sqlfilename = strcat('HR_',DataType,'.sqlite');
    elseif strcmp(DataType, 'valid')
        sqlfilename = strcat('HR_',DataType,'.sqlite');
    elseif strcmp(DataType, 'test')
        sqlfilename = strcat('HR_',DataType,'.sqlite');
    else
        error('Unknown data type to read, please select train, dev or test');
    end

    % Read data using SQLite
    sqlfile = fullfile('../HRInitialsClassificationData/', sqlfilename);
    conn = sqlite(sqlfile);
    data = fetch(conn, 'SELECT * FROM Data');
    
    % Process variables
    x1 = double(data.x1)';
    x2 = double(data.x2)';
    X = [x1; x2];
    y = double(data.y)';

    % One-hot encoding
    T = size(X, 2);
    yOneHot = zeros(3, T);
    for i=1:T
        yOneHot( y(i), i) = 1;
    end
end

