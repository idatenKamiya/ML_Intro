function [w, b, train_err, loo_err] = train_rls_loo(X, y, lambda)

    % Get dimensions
    [m, n] = size(X);
    % Display problem information
    fprintf('RLS with Leave-One-Out CV:\n');
    fprintf('  - Number of datapoints: %d, Features: %d\n', m, n);
    fprintf('  - Lambda: %.6f\n', lambda);
    
    %%First find RLS solution w_tilde = inv(C)*X'y
    fprintf('=====First training with simple RLS=====\n');
    %X = randn(100,3);
    %y = randn(100,1);
    %m = 100; n = 3; lambda = 0.01;
    % X̃ = [1, X] (add bias column)
    X_tilde = [ones(m, 1), X];  % m x (n+1)
    
    % Create regularization matrix Ĩ
    % ε=0, only weights (λ)
    I_tilde = diag([0, lambda * ones(1, n)]);  % (n+1) x (n+1)
    
    % Compute C = X̃ᵀX̃ + λĨ
    C = X_tilde' * X_tilde + I_tilde;  % (n+1) x (n+1)
    
    % d = X̃ᵀy
    d = X_tilde' * y;  % (n+1) x 1
    
    % Solve RLS: Cw̃ = d
    cond_C = cond(C);
    fprintf('  - Condition number of C: %.2e\n', cond_C);
    
    if cond_C < 1e12
        w_tilde = inv(C) * d;
    else
        w_tilde = pinv(C) * d;
    end
    
    % Extract bias and weights
    b = w_tilde(1);        % First element is bias
    w = w_tilde(2:end);    % Remaining elements are weights
    %b
    %w
    
    %% Compute Training Error (train_err)
    
    % y_pred
    y_train_pred = X_tilde * w_tilde;  % m x 1

    train_residuals = y - y_train_pred;
    train_err = mean(train_residuals.^2);
    
    fprintf('   →Simple RLS Training MSE: %.6f\n', train_err);
    
    %% Compute Leave-One-Out Cross-Validation Error
    
    fprintf('=====Computing LOOCV error====\n');
    
    loo_residuals = zeros(m, 1);
    
    % Compute LOOCV residuals using the derived formula:
    % residual_i = (w̃ᵀx̃ᵢ - yᵢ) / (1 - x̃ᵢᵀC⁻¹x̃ᵢ)
    %substitute alpha = x̃ᵢᵀC⁻¹x̃ᵢ
    
    for i = 1:m
        % Get i-th sample
        x_tilde_i = X_tilde(i, :)';  % (n+1) x 1 is column vector
        
        % alpha = x̃ᵢᵀC⁻¹x̃ᵢ
        alpha = x_tilde_i' * inv(C) * x_tilde_i;

        % check if scalar
        %alpha
        
        % Compute training residual for sample i
        train_residual_i = y_train_pred(i) - y(i);  % w̃ᵀx̃ᵢ - yᵢ
            
        % Apply LOOCV formula, residual_i = (w̃ᵀx̃ᵢ - yᵢ) / (1 - x̃ᵢᵀC⁻¹x̃ᵢ)
        loo_residuals(i) = train_residual_i / (1 - alpha);
    end
    
    % Compute MSE LOOCV error (loo_err)
    loo_err = mean(loo_residuals.^2);
    
    fprintf('   →LOOCV MSE: %.6f\n', loo_err);
    
    %Stats
    fprintf('\n=== Summary ===\n');
    fprintf('Training Error(train_err):     %.6f\n', train_err);
    fprintf('LOOCV Error(loo_err):        %.6f\n', loo_err);
end