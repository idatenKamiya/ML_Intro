%BU ID = skandan
clear all; clc;

% generate test data
X = randn(100, 3);
y = randn(100, 1);
lambda = 0.1;
epsilon = 0.01;

% Test both methods
[w1, b1] = train_rls(X, y, lambda, epsilon);
[w2, b2] = incremental_train_rls(X, y, lambda, epsilon);

% Compare results from rls and incremental rls
fprintf('RLS: w=[%.6f, %.6f, %.6f], b=%.6f\n', w1, b1);
fprintf('Incremental RLS: w=[%.6f, %.6f, %.6f], b=%.6f\n', w2, b2);

abs_diff_matrix = abs([w1; b1] - [w2; b2]);
fprintf('Absolute differences matrix:\n');
fprintf('[');
disp(abs_diff_matrix);
fprintf(']\n');