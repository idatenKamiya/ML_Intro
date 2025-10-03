clear all
close all

load adult_train_test

for k=1:10
    %shuffle training set
    idx=randperm(numel(ytrain));
    Xtrain=Xtrain(idx,:);
    ytrain=ytrain(idx);
    
    % train percetron (1 pass over training set)
    [w,b,average_w,average_b]=train_perceptron(Xtrain,ytrain);
    
    % test
    test_err_last_array(k)=numel(find(ytest~=sign(Xtest*w+b)))/numel(ytest);
    test_err_average_array(k)=numel(find(ytest~=sign(Xtest*average_w+average_b)))/numel(ytest);
end

test_err_last_array
test_err_average_array

T = table((1:10)', test_err_last_array', test_err_average_array', ...
    'VariableNames', {'Run', 'Last_Error', 'Average_Error'});

disp(T);


%Results:
%{ Run    Last_Error    Average_Error
%    1      0.2409        0.1549
%    2      0.2114        0.1494
%    3      0.1989        0.1539
%    4      0.1944        0.1534
%    5      0.1914        0.1569
%    6      0.1904        0.1559
%    7      0.1854        0.1534
%    8      0.1744        0.1549
%    9      0.1699        0.1549
%   10      0.1989        0.1564

%{ Observations:  
%   1. the averaged Perceptron performs better than the last solution
%   2. averaging the values smoothens the distribution and the last values
%}      tend to be sensitive to distribution pattern(variance)