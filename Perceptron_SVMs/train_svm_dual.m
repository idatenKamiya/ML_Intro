%%BU ID: skandan
%BU email: skandan@bu.edu

function [w, b] = train_svm_dual(X, y, lambda)
% TRAIN_SVM_DUAL Trains a soft-margin SVM in the dual form using quadprog
% Inputs:
%  X : m x d matrix of input features
%  y : m x 1 vector of labels (-1 or 1)
%  lambda : regularization hyperparameter
% Outputs:
%  w : d x 1 weight vector
%  b : bias

    [m, d] = size(X);

    %Step1 Compute Q = (1/(2*lambda)) * (y*y') .* (X*X')
    Y = y(:);
    Q = (1/(2*lambda)) * ( (Y*Y') .* (X*X') );

    f = -ones(m,1);  % linear term
    
    % Equality constraint: sum(alpha_i * y_i) = 0
    Aeq = Y'; %(1 x m)
    beq = 0;

    % Bounds: 0 <= alpha_i <= 1/m
    lb = zeros(m,1);
    ub = ones(m,1)/m;

    options = optimoptions('quadprog','Display','off');
    alpha = quadprog(Q, f, [], [], Aeq, beq, lb, ub, [], options);

    %Steps2 Compute primal weights w = sum(alpha_i * x_i * y_i)
    w = (1/(2*lambda)) * (X' * (alpha .* y));

    % Compute bias using given formula
    idx = find(alpha > 1e-5 & alpha < 1/m - 1e-5);
    b = median(y(idx) - X(idx,:) * w);
end
