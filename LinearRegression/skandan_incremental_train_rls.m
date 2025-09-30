%BU ID = skandan
%*_npo notation is used to denote (n+1) samples

function [w_npo, b_npo] = incremental_train_rls(X, y, lambda, epsilon)
    [m, n] = size(X);
    fprintf('training with incremental_train_rls):\n');
    fprintf('   →Structure of X - Features: %d, Lambda: %.6f, Epsilon: %.6f\n', n, lambda, epsilon);
    
    %Solution for RLS is => C*w_tilde=X'*y
    % create regularization matrix Q
    Q = diag([epsilon, lambda * ones(1, n)]);
    C_npo = Q; %C=(X'X+Q) => C=0+Q => C=Q when starting with 0 samples
    rhs_npo = zeros(n + 1, 1); %rhs=X'*y
    
    % process samples incrementally
    for i = 1:m
        % Get current sample
        x_i = X(i, :)';
        y_i = y(i);
        
        % create augmented feature vector: x_tilde_i = [1; x_i]
        x_tilde_i = [1; x_i];
        
        % incremental updates
        C_npo = C_npo + x_tilde_i * x_tilde_i';
        rhs_npo = rhs_npo + x_tilde_i * y_i;
    end
    
    % Solve final system
    cond_C_final = cond(C_npo);    
    if cond_C_final < 1e12
        fprintf('   ※C is iversible!\n');
        w_tilde_npo = inv(C_npo)* rhs_npo;
    else
        fprintf('   ※C is not iversible. Using pseudoinverse\n');
        w_tilde_npo = pinv(C_npo) * rhs_npo;
    end
    
    % get bias and weights
    b_npo = w_tilde_npo(1);
    w_npo = w_tilde_npo(2:end);
    
    fprintf('   →Final bias(b_npo): %.6f\n', b_npo);
    fprintf('   →Final Weights(w_npo): [%s]\n', sprintf('%.6f ', w_npo));
    
    % final training error
    X_tilde = [ones(m, 1), X];
    y_pred_npo = X_tilde * w_tilde_npo;
    mse_npo = mean((y - y_pred_npo).^2);
    fprintf('  ⇒Final training MSE: %.6f\n', mse_npo);
end