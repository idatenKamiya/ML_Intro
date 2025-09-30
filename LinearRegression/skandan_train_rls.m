%BU ID = skandan
function [w, b] = train_rls(X, y, lambda, epsilon)
    % Get dimensions
    [m, n] = size(X);
    fprintf('training with train_rls):\n');
    fprintf('   →Structure of X - Examples: %d, Features: %d\n', m, n);
    fprintf('   →Lambda: %.6f, Epsilon: %.6f\n', lambda, epsilon);

    %solution for RLS => w_tilde = inv(C) * X_tilde^T * y
    %X_tilde = [1, X] adding a bias of 1 to make augmented matrix
    X_tilde = [ones(m, 1), X];
    %regularization matrix Q
    Q = diag([epsilon, lambda * ones(1, n)]);
    %C = X_tilde^T * X_tilde + Q
    C = X_tilde' * X_tilde + Q;
    
    Xty = X_tilde' * y;
    
    % Check if C is invertible
    cond_C = cond(C);    
    if cond_C < 1e12
        fprintf('   ※C is invertible!\n');
        w_tilde = inv(C) * Xty;
    else
        fprintf('   ※C is not invertible. Using pseudoinverse\n');
        w_tilde = pinv(C) * Xty;
    end
    disp(w_tilde);
    % bias and weights
    b = w_tilde(1);
    w = w_tilde(2:end);
    fprintf('   →Bias: %.6f\n', b);
    fprintf('   →Weights: [%s]\n', sprintf('%.6f ', w));    
    % MSE
    y_pred = X_tilde * w_tilde;
    mse = mean((y - y_pred).^2);
    fprintf('   ⇒Training MSE: %.6f\n', mse);
end

%X = randn(100, 3);
%y = randn(100, 1);
% Train the model using the RLS algorithm
%[w, b] = train_rls(X, y, 0.1, 1e-3);