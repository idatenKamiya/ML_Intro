%BUID : skandan@bu.edu
%problem 4.2(a)

function [W] = train_svm_mhinge_sgd(Xtr, ytr, k, lambda, T)
    % Xtr: m x d
    % ytr: m x 1, values in 1..k
    % k: number of classes
    % lambda: regularization
    % T: number of iterations
    
    [m, d] = size(Xtr);
    
    % Initialize weight matrix
    W = zeros(d, k);
    
    for t = 1:T
        % Sample one training example
        i = randi(m);
        x_i = Xtr(i, :)';    % d x 1
        y_i = ytr(i);        % true label
        
        % Compute scores for all classes
        scores = W' * x_i;   % k x 1
        
        % Find the class with highest score != y_i
        scores(y_i) = -Inf;  % exclude true class
        [max_score, y_hat] = max(scores); % y_hat != y_i
        
        % Compute hinge loss
        loss = 1 + max_score - W(:, y_i)' * x_i;
        
        % Initialize stochastic gradient
        grad = lambda * W; % d x k
        
        if loss > 0
            % Update columns corresponding to y_i and y_hat
            grad(:, y_i) = grad(:, y_i) - x_i;
            grad(:, y_hat) = grad(:, y_hat) + x_i;
        end
        
        % Learning rate
        eta = 1 / (lambda * t);
        
        % Update W
        W = W - eta * grad;
    end
end
