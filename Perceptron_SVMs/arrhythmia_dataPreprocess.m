% BU ID: skandan
% BU email: skandan@bu.edu
% Problem 3.3(a) - file to preprocess data
% CLean and Randomly split the data in 80-20 for train-test

% Load the arrhythmia data file
load('arrhythmia.mat');

% initial data information
fprintf('Original data dimensions:\n');
fprintf('X: %d samples x %d features\n', size(X)); %(452x279)
fprintf('Y: %d labels\n', length(Y)); %(452x1)
% Checing class distribution in Y
unique_labels = unique(Y);
fprintf('\nOriginal label distribution:\n');
fprintf('Class 0 (false): %d samples (%.2f%%)\n', sum(Y == false), mean(Y == false) * 100);
fprintf('Class 1 (true):  %d samples (%.2f%%)\n', sum(Y == true), mean(Y == true) * 100);

% Convert logical labels to 0 and 1
y = double(Y); % Convert logical to double (0 and 1)

%% Imputation
% Impute missing values with median of each feature
X_imputed = X; % Create a copy of the original data

for i = 1:size(X, 2)
    col_data = X(:, i);
    if any(isnan(col_data))
        median_val = median(col_data, 'omitnan');
        X_imputed(isnan(col_data), i) = median_val;
    end
end

%% Split data by class for sampling (80% train, 20% test)
class0_idx = find(y == 0);
class1_idx = find(y == 1);

fprintf('\nClass distribution for splitting:\n');
fprintf('Class 0: %d samples\n', length(class0_idx));
fprintf('Class 1: %d samples\n', length(class1_idx));

% Set random seed for reproducibility
rng(42);

% Shuffle indices within each class
class0_idx = class0_idx(randperm(length(class0_idx)));
class1_idx = class1_idx(randperm(length(class1_idx)));

% Calculate split points (80% train, 20% test) for each class
train_ratio = 0.8;

% Create index counters
class0_train_count = round(length(class0_idx) * train_ratio);
class1_train_count = round(length(class1_idx) * train_ratio);

% Split class 0
class0_train_idx = class0_idx(1:class0_train_count);
class0_test_idx = class0_idx(class0_train_count+1:end);

% Split class 1  
class1_train_idx = class1_idx(1:class1_train_count);
class1_test_idx = class1_idx(class1_train_count+1:end);

% Combine train and test indices
train_idx = [class0_train_idx; class1_train_idx];
test_idx = [class0_test_idx; class1_test_idx];

% Shuffle the combined indices to mix classes
train_idx = train_idx(randperm(length(train_idx)));
test_idx = test_idx(randperm(length(test_idx)));

%% Create training and test sets
X_train = X_imputed(train_idx, :);
y_train = y(train_idx);
X_test = X_imputed(test_idx, :);
y_test = y(test_idx);

%% Save the processed data
save('arrhythmia_train_test.mat', 'X_train', 'y_train', 'X_test', 'y_test', 'train_idx', 'test_idx', 'X_imputed', 'y');