%BUID : skandan@bu.edu
%problem 4.2(b)

function [ypred] = test_svm_multi(W, Xte)
% Function is used to Predict labels for multiclass linear SVM
%
%   ypred = test_svm_multi(W, Xte)
%
%   INPUT:
%       W   - d x k weight matrix learned from training
%       Xte - m_test x d matrix of test samples (or validation samples too)
%
%   OUTPUT:
%       ypred - m_test x 1 vector of predicted class labels (1..k)

    % Compute scores for all test examples
    % Xte: m_test x d
    % W:   d x k
    % scores: m_test x k
    scores = Xte * W;

    % Pick the class with the maximum score for each test example
    % max(A, row, column) takes 3 arguements. returns the (max value,
    % index). Up to us to choose row-wise max value.
    [~, ypred] = max(scores, [], 2);


end
