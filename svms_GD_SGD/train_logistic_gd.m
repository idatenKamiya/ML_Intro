%BUID : skandan@bu.edu
%problem 4.1(b)

function [w, obj] = train_logistic_gd(Xtr, ytr, lambda, T)
    % Get the number of features
    [m, d] = size(Xtr);
    
    % Initialize weights to zero
    w = zeros(d, 1);
    
    % Initialize objective value vector
    obj = zeros(T, 1);
    
    for t = 1:T
        % Compute the objective at the beginning of this iteration
        z = ytr .* (Xtr * w);                      % m x 1
        logistic_loss = log(1 + exp(-z));          % m x 1
        obj(t) = (lambda / 2) * (w' * w) + mean(logistic_loss);
        
        % Compute the gradient of F(w)
        grad = lambda * w - (1/m) * (Xtr' * (ytr ./ (1 + exp(z))));
        
        % Learning rate: eta_t = 1 / (lambda * t)
        eta = 1 / (lambda * t);
        
        % Gradient descent update
        w = w - eta * grad;
    end
end
