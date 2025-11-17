%BU ID: skandan
% HW7 Problem 2.1(4) the empirical risk minimizer: computing
% missclassification (test-error evaluation)

clc; clear; rng('shuffle');

% Parameters
N = 10;                     % number of intervals
grid_points = (0:N)/N;             % discretization x0...xN
H = [];                     % hypothesis class [i,j]
for i = 0:N-1
    for j = i+1:N
        H = [H; i j];
    end
end
numH = size(H,1);           % |H| = 55

m = 100;                    % training sample size
test_points = 10000;        % test set size
T = 200;                     % number of trials
eps = 0.05;                 % optional epsilon (for reference)

% Storage for test errors
test_error_frac = zeros(T,1);

%% Main loop: run trials
for t = 1:T
    % pick true interval f*
    k = randi(numH);
    i_star = H(k,1); j_star = H(k,2);
    a_star = grid_points(i_star+1); b_star = grid_points(j_star+1);

    % training sample
    xs_train = rand(m,1);
    ys_train = (xs_train >= a_star) & (xs_train <= b_star);

    % ERM: brute-force search over H
    bestErr = m+1;
    bestLen = -1;
    bestIdx = -1;
    for hidx = 1:numH
        i = H(hidx,1); j = H(hidx,2);
        a = grid_points(i+1); b = grid_points(j+1);
        h_pred = (xs_train >= a) & (xs_train <= b);
        err = sum(h_pred ~= ys_train);
        len = j - i;
        if err < bestErr || (err == bestErr && len > bestLen)
            bestErr = err;
            bestLen = len;
            bestIdx = hidx;
        end
    end

    % Selected hypothesis
    i_sel = H(bestIdx,1); j_sel = H(bestIdx,2);
    a_sel = grid_points(i_sel+1); b_sel = grid_points(j_sel+1);

    % Test set
    xs_test = rand(test_points,1);
    ys_test = (xs_test >= a_star) & (xs_test <= b_star);
    preds_test = (xs_test >= a_sel) & (xs_test <= b_sel);

    % Fraction of misclassified test points
    test_error_frac(t) = mean(preds_test ~= ys_test);
end

% Results
mean_test_error = mean(test_error_frac);
std_test_error = std(test_error_frac);

fprintf('After %d trials with m=%d training points:\n', T, m);
fprintf('Mean test-set misclassification fraction = %.4f\n', mean_test_error);
fprintf('Std of test-set misclassification fraction = %.4f\n', std_test_error);

% plot histogram of test errors
figure;
histogram(test_error_frac, 0:0.01:1, 'FaceColor', [0 0.5 1]);
xlabel('Test-set misclassification fraction');
ylabel('Number of trials');
title(sprintf('N=%d, m=%d, T=%d trials', N, m, T));
grid on;
