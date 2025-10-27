%BUID : skandan@bu.edu
%problem 4.2(c) & 4.2(d)
% traintest_svm_mnist.m
clear; clc; close all;

%Load MNIST data
load('mnist.mat');  % contains Xtr, ytr, Xval, yval, Xte, yte

% Ensure matrices are full to make SGD work faster
Xtr = full(Xtr); 
Xval = full(Xval);
Xte = full(Xte);

% Number of classes
k = max([ytr; yval; yte]);

%Candidate lambdas for model selection
lambdas = [0.0001, 0.001, 0.01, 0.1, 1];

% Number of SGD iterations
T = 1e6;

best_acc = 0;
best_lambda = 0;

%Model selection using validation set
fprintf('Selecting best lambda using validation set...\n');
for i = 1:length(lambdas)
    lambda = lambdas(i);
    fprintf('Training with lambda = %.5f ...\n', lambda);
    
    % Train on training set only
    W = train_svm_mhinge_sgd(Xtr, ytr, k, lambda, T);
    
    % Predict on validation set
    yval_pred = test_svm_multi(W, Xval);
    
    % Compute accuracy
    acc = mean(yval_pred == yval) * 100;
    fprintf('Validation accuracy: %.2f%%\n', acc);
    
    % Keep track of best lambda
    if acc > best_acc
        best_acc = acc;
        best_lambda = lambda;
    end
end

fprintf('Best lambda selected: %.5f with validation accuracy %.2f%%\n', best_lambda, best_acc);

%% Retrain on training + validation sets with best lambda
fprintf('Retraining on training + validation set with lambda = %.5f ...\n', best_lambda);

Xtrain_full = [Xtr; Xval];
ytrain_full = [ytr; yval];

W_final = train_svm_mhinge_sgd(Xtrain_full, ytrain_full, k, best_lambda, T);

%% Test on the test set
fprintf('Testing on the test set...\n');
ytest_pred = test_svm_multi(W_final, Xte);
test_acc = mean(ytest_pred == yte) * 100;
fprintf('Test set accuracy: %.2f%%\n', test_acc);

% Test error (0/1 loss)
test_error = mean(ytest_pred ~= yte);  % fraction of misclassified examples
fprintf('Test error (0/1 loss): %.4f\n', test_error);

%% Confusion matrix
CM = confusionmat(yte, ytest_pred);  % rows: true classes, cols: predicted classes

% Rearrange so rows = predicted, columns = true (as requested)
CM = CM';  

disp('Confusion matrix (rows = predicted, columns = true):');
disp(CM);
