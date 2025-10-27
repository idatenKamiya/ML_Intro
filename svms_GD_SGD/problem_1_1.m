%BU ID: skandan@bu.edu
%problem  1.1
%2D plot of the anomaly classifier

% One-Class SVM style anomaly detector visualization in 2D
% Example 1: Hard separation
% Example 2: Soft-margin with some points on the anomaly side

clear; clc; close all;

%% --------------------------
%  Example 1: Hard-Margin
% ---------------------------
w1 = [1; 0.5];  % weight vector

x1_vals = linspace(-3, 4, 200);
x2_vals = (1 - w1(1)*x1_vals) / w1(2); % decision boundary line

% Normal points (strictly satisfy w^T x >= 1)
normal_points1 = [ 2.0  0.0;
                   1.5  1.0;
                   0.5  2.0;
                   3.0 -1.0 ];

% Anomaly points (w^T x < 1)
anomaly_points1 = [-1.0 -1.0;
                    0.0  0.0;   % point of interest
                    0.5 -1.0;
                   -2.0  1.0];

%figure1 shows plot with example1
figure(1); hold on; grid on;
title('Figure 1: One-Class Linear Classifier (Hard-Margin)');
xlabel('x_1'); ylabel('x_2');

% Shade anomalous region
if w1(2) > 0
    fill([x1_vals fliplr(x1_vals)], [x2_vals fliplr(x2_vals)-10], ...
         [0.9 0.9 0.9], 'EdgeColor', 'none');
else
    fill([x1_vals fliplr(x1_vals)], [x2_vals fliplr(x2_vals)+10], ...
         [0.9 0.9 0.9], 'EdgeColor', 'none');
end

% Decision boundary
plot(x1_vals, x2_vals, 'k-', 'LineWidth', 1.5);

% Points
plot(normal_points1(:,1), normal_points1(:,2), 'bo', 'MarkerSize', 8, 'LineWidth', 1.5);
plot(anomaly_points1(:,1), anomaly_points1(:,2), 'rx', 'MarkerSize', 8, 'LineWidth', 1.5);

% Axes + labels
text(0.2, -0.4, '(0,0)', 'FontSize', 10);
xlim([-3 4]); ylim([-3 3]);
line([0 0], ylim, 'Color', [0.5 0.5 0.5], 'LineStyle', '--');
line(xlim, [0 0], 'Color', [0.5 0.5 0.5], 'LineStyle', '--');
legend('Decision boundary w^T x = 1', 'Anomalous region', 'Normal', 'Anomaly', 'Location', 'best');
hold off;


%% --------------------------
%  Example 2: Soft-Margin
% ---------------------------
% New weight vector to make some normals fall inside anomaly region
w2 = [0.8; 0.8];

x1_vals2 = linspace(-3, 4, 200);
x2_vals2 = (1 - w2(1)*x1_vals2) / w2(2);

% Normal points - some deliberately placed inside anomaly side
normal_points2 = [ 1.0  1.0;   % this is on the boundary
                   0.5  0.5;   % this is inside anomaly region
                   2.0  0.0;
                   1.0  2.0 ];

% Anomalies
anomaly_points2 = [ -1.0 -1.0;
                     0.0  0.0;
                     0.5 -1.0 ];

% figure2 shows plot for example2
figure(2); hold on; grid on;
title('Figure 2: One-Class Linear Classifier (Soft-Margin Case)');
xlabel('x_1'); ylabel('x_2');

% Shade anomalous region
if w2(2) > 0
    fill([x1_vals2 fliplr(x1_vals2)], [x2_vals2 fliplr(x2_vals2)-10], ...
         [0.9 0.9 0.9], 'EdgeColor', 'none');
else
    fill([x1_vals2 fliplr(x1_vals2)], [x2_vals2 fliplr(x2_vals2)+10], ...
         [0.9 0.9 0.9], 'EdgeColor', 'none');
end

% Decision boundary
plot(x1_vals2, x2_vals2, 'k-', 'LineWidth', 1.5);

% Plot normal and anomaly points
plot(normal_points2(:,1), normal_points2(:,2), 'bo', 'MarkerSize', 8, 'LineWidth', 1.5);
plot(anomaly_points2(:,1), anomaly_points2(:,2), 'rx', 'MarkerSize', 8, 'LineWidth', 1.5);

% Annotate the "violating" normal point
text(0.5+0.1, 0.5-0.3, 'Normal point inside anomaly region', 'FontSize', 9, 'Color', 'b');

% Axes and legend
xlim([-3 4]); ylim([-3 3]);
line([0 0], ylim, 'Color', [0.5 0.5 0.5], 'LineStyle', '--');
line(xlim, [0 0], 'Color', [0.5 0.5 0.5], 'LineStyle', '--');
legend('Decision boundary w^T x = 1', 'Anomalous region', 'Normal', 'Anomaly', 'Location', 'best');
hold off;
