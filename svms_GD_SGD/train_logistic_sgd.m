%BUID : skandan@bu.edu
%problem 4.1(d)

function [w, obj] = train_logistic_sgd(Xtr, ytr, lambda, T)
    % Get number of features
    [m, d] = size(Xtr);
    
    % Initialize weights to zero
    w = zeros(d, 1);
    
    % Initialize objective vector
    obj = zeros(T, 1);
    
    for t = 1:T
        % Compute full objective at the beginning of this iteration
        z = ytr .* (Xtr * w);                      % m x 1
        logistic_loss = log(1 + exp(-z));          % m x 1
        obj(t) = (lambda / 2) * (w' * w) + mean(logistic_loss);
        
        % Sample one training example uniformly at random
        i = randi(m);                              % random index
        x_i = Xtr(i, :)';                          % column vector
        y_i = ytr(i);
        
        % Compute stochastic gradient
        g_stoch = lambda * w - (y_i * x_i) / (1 + exp(y_i * (w' * x_i)));
        
        % Learning rate
        eta = 1 / (lambda * t);
        
        % Update weights
        w = w - eta * g_stoch;
    end
end
