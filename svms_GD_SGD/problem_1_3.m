%BUID : skandan@bu.edu
%problem1.3

clear; clc; close all;

%% Parameters
gamma = 1;           % RBF kernel parameter
train_X = [ 1  1;
            1 -1;
           -1  1;
           -1 -1 ];   % 4 training points in R^2
n = size(train_X, 1);
alpha = ones(n,1) / n;  % α_i = 1/n

%% Create grid over which to evaluate f(z)
xrange = linspace(-3, 3, 200);
yrange = linspace(-3, 3, 200);
[X, Y] = meshgrid(xrange, yrange);

% Flatten grid for easier vectorized computation
Z = [X(:) Y(:)];
fvals = zeros(size(Z,1),1);

%% Compute f(z) for each grid point
for i = 1:size(Z,1)
    z = Z(i,:);
    % Sum over training points
    val = 0;
    for j = 1:n
        diff = z - train_X(j,:);
        dist2 = diff*diff';  % squared Euclidean distance
        val = val + alpha(j) * exp(-gamma * dist2);
    end
    fvals(i) = val;
end

% Reshape back to grid
F = reshape(fvals, size(X));

%% Evaluate f(0,0) and check anomaly
z0 = [0 0];
f_z0 = 0;
for j = 1:n
    diff = z0 - train_X(j,:);
    dist2 = diff*diff';
    f_z0 = f_z0 + alpha(j) * exp(-gamma * dist2);
end

fprintf('f(0,0) = %.6f\n', f_z0);
if f_z0 < 1
    fprintf('Point (0,0) is OUTSIDE the decision boundary → flagged as ANOMALOUS.\n');
else
    fprintf('Point (0,0) is INSIDE the decision boundary → flagged as NORMAL.\n');
end

%% Plot contour map
figure; hold on; grid on;
contourf(X, Y, F, 30, 'LineColor', 'none');    % heat map of f(z)
colorbar;
title('Kernel Anomaly Detector f(z) with RBF kernel (\gamma = 1)');
xlabel('x_1'); ylabel('x_2');

% Plot contour line for f(z) = 1
contour(X, Y, F, [1 1], 'k', 'LineWidth', 1.5);

% Plot training points
plot(train_X(:,1), train_X(:,2), 'wo', 'MarkerSize', 8, 'LineWidth', 1.5);

% Mark the point (0,0)
plot(0, 0, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
text(0.2, -0.3, '(0,0)', 'Color', 'r');

legend('f(z)', 'f(z)=1 boundary', 'Training points', '(0,0)', 'Location', 'best');

xlim([-3 3]); ylim([-3 3]);
axis equal
hold off;
