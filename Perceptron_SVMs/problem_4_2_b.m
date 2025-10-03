% BU ID: skandan
% BU email: skandan@bu.edu
% problem 4.2(b)

% Laod preprocessed data
load('arrhythmia_train_test.mat');  % X_train, ytrain, X_test, ytest

%% Convert logical labels to numeric [1 -1]
y_train = double(y_train); %ensuring logical is converted to numerical
y_train(y_train==0) = -1;
y_test = double(y_test);
y_test(y_test==0) = -1;

%% Hyperparameter grids
C_grid = 2.^(-2:4);        % BoxConstraint values (kambda)
gamma_grid = 2.^(-10:0);   % RBF kernel scale
poly_degree = [1,2,3];     % Polynomial degrees

kfold = 3;
cv = cvpartition(y_train,'KFold',kfold,'Stratify',true);

% -----------------------------
% 1. Linear SVM
% -----------------------------
best_C = C_grid(1);
best_acc = 0;

for C = C_grid
    acc_fold = zeros(kfold,1);
    for k = 1:kfold
        Xtr = X_train(training(cv,k),:);
        ytr = y_train(training(cv,k));
        Xval = X_train(test(cv,k),:);
        yval = y_train(test(cv,k));

        if numel(unique(ytr)) < 2
            acc_fold(k) = NaN; % skip fold
            continue;
        end

        SVMModel = fitcsvm(Xtr, ytr, 'KernelFunction','linear', 'BoxConstraint', C);
        ypred = predict(SVMModel, Xval);
        acc_fold(k) = mean(ypred == yval);
    end
    mean_acc = nanmean(acc_fold);
    if mean_acc > best_acc
        best_acc = mean_acc;
        best_C = C;
    end
end

fprintf('-------------------Results--------------------');
fprintf('Linear SVM best C: %.5f, CV acc: %.4f\n', best_C, best_acc);

% Final fit
if numel(unique(y_train)) < 2
    error('Training set contains only one class! Cannot train SVM.');
end
linearSVM = fitcsvm(X_train, y_train, 'KernelFunction','linear', 'BoxConstraint', best_C);
ytest_pred = predict(linearSVM, X_test);
fprintf('Linear SVM test accuracy: %.4f\n', mean(ytest_pred == y_test));
fprintf('-----------------------------------\n');

% -----------------------------
% 2. Gaussian (RBF) SVM
% -----------------------------
best_C = C_grid(1);
best_gamma = gamma_grid(1);
best_acc = 0;

for C = C_grid
    for gamma = gamma_grid
        acc_fold = zeros(kfold,1);
        for k = 1:kfold
            Xtr = X_train(training(cv,k),:);
            ytr = y_train(training(cv,k));
            Xval = X_train(test(cv,k),:);
            yval = y_train(test(cv,k));

            if numel(unique(ytr)) < 2
                acc_fold(k) = NaN;
                continue;
            end

            SVMModel = fitcsvm(Xtr, ytr, 'KernelFunction','rbf','KernelScale', 1/sqrt(2*gamma), 'BoxConstraint', C);
            ypred = predict(SVMModel, Xval);
            acc_fold(k) = mean(ypred == yval);
        end
        mean_acc = nanmean(acc_fold);
        if mean_acc > best_acc
            best_acc = mean_acc;
            best_C = C;
            best_gamma = gamma;
        end
    end
end

fprintf('Gaussian SVM best C: %.5f, gamma: %.5f, CV acc: %.4f\n', best_C, best_gamma, best_acc);
gaussianSVM = fitcsvm(X_train, y_train, 'KernelFunction','rbf','KernelScale', 1/sqrt(2*best_gamma), 'BoxConstraint', best_C);
ytest_pred = predict(gaussianSVM, X_test);
fprintf('Gaussian SVM test accuracy: %.4f\n', mean(ytest_pred == y_test));
fprintf('-----------------------------------\n');

% -----------------------------
% 3. Polynomial SVM
% -----------------------------
best_C = C_grid(1);
best_degree = poly_degree(1);
best_acc = 0;

for C = C_grid
    for deg = poly_degree
        acc_fold = zeros(kfold,1);
        for k = 1:kfold
            Xtr = X_train(training(cv,k),:);
            ytr = y_train(training(cv,k));
            Xval = X_train(test(cv,k),:);
            yval = y_train(test(cv,k));

            if numel(unique(ytr)) < 2
                acc_fold(k) = NaN;
                continue;
            end

            SVMModel = fitcsvm(Xtr, ytr, 'KernelFunction','polynomial', ...
                               'PolynomialOrder', deg, 'BoxConstraint', C);
            ypred = predict(SVMModel, Xval);
            acc_fold(k) = mean(ypred == yval);
        end
        mean_acc = nanmean(acc_fold);
        if mean_acc > best_acc
            best_acc = mean_acc;
            best_C = C;
            best_degree = deg;
        end
    end
end

fprintf('Polynomial SVM best C: %.5f, degree: %d, CV acc: %.4f\n', best_C, best_degree, best_acc);
polySVM = fitcsvm(X_train, y_train, 'KernelFunction','polynomial','PolynomialOrder', best_degree, 'BoxConstraint', best_C);
ytest_pred = predict(polySVM, X_test);
fprintf('Polynomial SVM test accuracy: %.4f\n', mean(ytest_pred == y_test));

