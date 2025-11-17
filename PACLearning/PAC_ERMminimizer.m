%BU ID: skandan
% HW7 Problem 2.1(3) the empirical risk minimizer: the interval [xi, xj ] 
% that minimizes the training error.

clc; clear;
% PAC simulation for N=10 intervals
rng('shuffle');

N = 10;
grid_points = (0:N)/N;            % x0...xN
H = [];                    % store all candidate intervals as [i j]
for i=0:N-1
    for j=i+1:N
        H = [H; i j];
    end
end
numH = size(H,1);          % should be 55 for 10 points

% experiment parameters
eps = 0.05;                % epsilon tolerance
delta = 0.05;              % desired delta (for reference)
m_values = [10,20,40,80,160,320];  % sample sizes to test
T = 2000;                  % trials per m

% Precompute theoretical PAC m for reference (natural log)
theoretical_m = (1/eps) * (log(numH) + log(1/delta));
fprintf('N=%d, |H|=%d, PAC-theoretical m (given form) = %.1f\n', N, numH, theoretical_m);

emp_failure = zeros(size(m_values));

for mm = 1:length(m_values)
    m = m_values(mm);
    failures = 0;
    for t = 1:T
        % sample true interval uniformly from H
        k = randi(numH);
        i_star = H(k,1);
        j_star = H(k,2);
        a_star = grid_points(i_star+1); % +1 because MATLAB indexing!!!!!!!!!!!!!!!
        b_star = grid_points(j_star+1);

        % draw m sample x's uniformly
        xs = rand(m,1);
        ys = (xs >= a_star) & (xs <= b_star);  % true labels

        % find ERM: brute force over all hypotheses
        bestErr = m+1;
        bestLen = -1;
        bestIdx = -1;
        for hidx = 1:numH
            i = H(hidx,1); j = H(hidx,2);
            h_pred = (xs >= grid_points(i+1)) & (xs <= grid_points(j+1));
            err = sum(h_pred ~= ys);
            len = j - i;  % length in grid steps
            if err < bestErr || (err == bestErr && len > bestLen)
                bestErr = err;
                bestLen = len;
                bestIdx = hidx;
            end
        end

        % selected hypothesis
        i_sel = H(bestIdx,1); j_sel = H(bestIdx,2);
        a_sel = grid_points(i_sel+1); b_sel = grid_points(j_sel+1);

        % empirical error
        Lhat = bestErr / m;

        % true error L: length of symmetric difference
        len_h = (j_sel - i_sel)/N;
        len_star = (j_star - i_star)/N;
        inter_left = max(a_sel, a_star);
        inter_right = min(b_sel, b_star);
        inter_len = max(0, inter_right - inter_left);
        L = len_h + len_star - 2*inter_len;

        if abs(L - Lhat) > eps
            failures = failures + 1;
        end
    end

    emp_failure(mm) = failures / T;
    fprintf('m=%4d  empirical failure rate = %.4f\n', m, emp_failure(mm));
end

% plot
figure;
plot(m_values, emp_failure, '-o', 'LineWidth', 2);
xlabel('m');
ylabel('{Empirical failure rate} $P(|L-\hat{L}| > \epsilon)$', 'Interpreter', 'latex');

title(sprintf('N=%d, \\epsilon=%.3f, T=%d trials', N, eps, T));
grid on;
