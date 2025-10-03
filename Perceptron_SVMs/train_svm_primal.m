%%BU ID: skandan
%BU email: skandan@bu.edu

function [w, b] = train_svm_primal(X, y, lambda)
% TRAIN_SVM_PRIMAL Trains a soft-margin SVM in the primal form using quadprog
% Inputs:
%   X      : m x d matrix of input features
%   y      : m x 1 vector of labels (-1 or 1)
%   lambda : regularization parameter
% Outputs:
%   w : d x 1 weight vector
%   b : bias

    [m, d] = size(X);
    
    % Total number of variables: [w (d), b (1), xi (m)]
    nVars = d + 1 + m;
    
    % H matrix (quadratic term)
    H = zeros(nVars);
    H(1:d, 1:d) = 2*lambda * eye(d);  % Only w^T w term

    % f vector (linear term)
    f = zeros(nVars,1);
    f(d+2:end) = 1/m;  % coefficients for xi

    % Inequality constraints: y_i (w^T x_i + b) + xi_i >= 1
    % Convert to A*z <= b form
    A = zeros(m, nVars);
    for i = 1:m
        A(i,1:d) = -y(i) * X(i,:);
        A(i,d+1) = -y(i);    % bias
        A(i,d+1+i) = -1;     % xi_i
    end
    b_ineq = -ones(m,1);

    % Lower bounds (xi_i >= 0), others unbounded
    lb = [-inf(d+1,1); zeros(m,1)];
    ub = [];

    % Solve QP using Matlab's quadprog
    options = optimoptions('quadprog','Display','off');
    z = quadprog(H, f, A, b_ineq, [], [], lb, ub, [], options);

    % Extract w and b
    w = z(1:d);
    b = z(d+1);
end
