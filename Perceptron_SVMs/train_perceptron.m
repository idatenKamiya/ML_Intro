%BU ID: skandan
%BU email: skandan@bu.edu
%% train_perceptron module used to train the perceptron algorithm
function [w, b, average_w, average_b] = train_perceptron(X, y)

% Input:
%   X : m x d matrix of input samples (rows are samples)
%   y : m x 1 vector of labels in {-1, +1}
%
% Output:
%   w          : final weight vector (d x 1)
%   b          : final bias scalar
%   average_w  : averaged weight vector (d x 1)
%   average_b  : averaged bias scalar
    
    [m, d] = size(X);
    
    % Augment data with 1 for bias handling X = [1 X]
    X_aug = [X, ones(m,1)];
    
    % Initialize weight vector (including bias term) w = [b w]
    w_aug = zeros(d+1, 1);
    
    % For averaging
    w_sum = zeros(d+1, 1);

    counter = 0;
    % Number of passes through data (epochs)
    epoch = 10;  % You can adjust this
    
    for t = 1:epoch
        for i = 1:m
            xi = X_aug(i, :)';   % column vector
            yi = y(i);
            
            % Perceptron update rule
            if yi * (w_aug' * xi) <= 0
                w_aug = w_aug + yi * xi;
            end
            
            % Accumulate for averaging
            w_sum = w_sum + w_aug;
            counter = counter + 1;
        end
    end
    
    % Separate w and b
    w = w_aug(1:d);
    b = w_aug(end);
    
    % Averaged versions
    average_w = w_sum(1:d) / counter;
    average_b = w_sum(end) / counter;
end

