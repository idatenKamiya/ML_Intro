%BUID : skandan@bu.edu
%problem 4.1(e)

%% test_exact_stochastic.m
clear; clc; close all;

%% Generate a sample dataset
rng(0); % for reproducibility
m = 200;   % number of samples
d = 10;    % number of features

Xtr = randn(m, d);           % random features
true_w = randn(d, 1);        % true weight vector
ytr = sign(Xtr * true_w + 0.1*randn(m,1)); % labels with small noise

%% Parameters
lambda = 0.1;
T = 200; % number of iterations

%% Exact Gradient Descent
[w_gd, obj_gd] = train_logistic_gd(Xtr, ytr, lambda, T);

fprintf('Exact GD:\n');
fprintf('Final objective: %.6f\n', obj_gd(end));
fprintf('Norm of w: %.6f\n\n', norm(w_gd));

%% Plot for GD
figure;
plot(1:T, obj_gd, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Objective F(w)');
title('Exact Gradient Descent');
grid on;

%% Stochastic Gradient Descent
[w_sgd, obj_sgd] = train_logistic_sgd(Xtr, ytr, lambda, T);

fprintf('Stochastic GD:\n');
fprintf('Final objective: %.6f\n', obj_sgd(end));
fprintf('Norm of w: %.6f\n\n', norm(w_sgd));

%% Plot for SGD
figure;
plot(1:T, obj_sgd, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Objective F(w)');
title('Stochastic Gradient Descent');
grid on;

%% Optional: Compare GD vs SGD in one figure
figure;
plot(1:T, obj_gd, 'b-', 'LineWidth', 2); hold on;
plot(1:T, obj_sgd, 'r-', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Objective F(w)');
title('GD vs SGD');
legend('Exact GD','Stochastic GD');
grid on;
