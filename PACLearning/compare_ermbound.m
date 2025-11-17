%BU ID: skandan
% HW7 Problem 2.1(4) the empirical risk minimizer: compare PAC ERM against
% Test-error. Report results for Varying N and m values
% Compare PAC Bound vs Observed Test Error
clc; clear; rng('shuffle');

%% sweep over N or sweep over m (true = increasing N; false = increasing m)
sweepN = true;   % set to false to sweep over m instead

%% Parameters
eps = 0.05;       % epsilon for PAC bound
delta = 0.05;     % delta for PAC bound
T = 200;          % number of trials per setting
test_points = 10000;

% Increase N or m
if sweepN
    N_values = 5:5:50;   % sweep N
    m = 100;             % fixed m
else
    m_values = 10:10:200;  % sweep m
    N = 10;               % fixed N
end

mean_test_errors = [];
PAC_required_m = [];


%% Main loop
if sweepN
    for N = N_values
        fprintf('Running for N = %d...\n', N);

        grid_points = (0:N)/N; % interval [0, 1]

        % Build hypothesis class
        H = [];
        for i = 0:N-1
            for j = i+1:N
                H = [H; i j];
            end
        end
        numH = size(H,1); % (H is order[55x2])

        % PAC bound required m
        PAC_m = (1/eps) * ( log(numH) + log(1/delta) );
        PAC_required_m(end+1) = PAC_m;

        % Simulation to measure empirical test error
        test_error_frac = zeros(T,1);

        for t = 1:T
            % pick true interval
            k = randi(numH);
            i_star = H(k,1); j_star = H(k,2);
            a_star = grid_points(i_star+1); b_star = grid_points(j_star+1);

            % training set
            xs_train = rand(m,1);
            ys_train = (xs_train >= a_star) & (xs_train <= b_star);

            % ERM
            bestErr = m+1; bestLen = -1; bestIdx = -1;
            for hidx = 1:numH
                i = H(hidx,1); j = H(hidx,2);
                a = grid_points(i+1); b = grid_points(j+1);
                h_pred = (xs_train >= a) & (xs_train <= b);
                err = sum(h_pred ~= ys_train);
                len = j - i;
                if err < bestErr || (err == bestErr && len > bestLen)
                    bestErr = err; bestLen = len; bestIdx = hidx;
                end
            end

            % Selected hypothesis
            i_sel = H(bestIdx,1); j_sel = H(bestIdx,2);
            a_sel = grid_points(i_sel+1); b_sel = grid_points(j_sel+1);

            % Test set
            xs_test = rand(test_points,1);
            ys_test = (xs_test >= a_star) & (xs_test <= b_star);
            preds = (xs_test >= a_sel) & (xs_test <= b_sel);

            test_error_frac(t) = mean(preds ~= ys_test);
        end

        mean_test_errors(end+1) = mean(test_error_frac);
    end
else
    for m = m_values
        fprintf('Running for m = %d...\n', m);

        grid_points = (0:N)/N;

        % Build hypothesis class
        H = [];
        for i = 0:N-1
            for j = i+1:N
                H = [H; i j];
            end
        end
        numH = size(H,1);

        % PAC-required m for comparison
        PAC_required_m(end+1) = (1/eps)*(log(numH)+log(1/delta));

        test_error_frac = zeros(T,1);

        for t = 1:T
            % pick true hypothesis
            k = randi(numH);
            i_star = H(k,1); j_star = H(k,2);
            a_star = grid_points(i_star+1); b_star = grid_points(j_star+1);

            xs_train = rand(m,1);
            ys_train = (xs_train >= a_star) & (xs_train <= b_star);

            % ERM
            bestErr = m+1; bestLen = -1; bestIdx = -1;
            for hidx = 1:numH
                i = H(hidx,1); j = H(hidx,2);
                a = grid_points(i+1); b = grid_points(j+1);
                h_pred = (xs_train >= a) & (xs_train <= b);
                err = sum(h_pred ~= ys_train);
                len = j - i;
                if err < bestErr || (err == bestErr && len > bestLen)
                    bestErr = err; bestLen = len; bestIdx = hidx;
                end
            end

            i_sel = H(bestIdx,1); j_sel = H(bestIdx,2);
            a_sel = grid_points(i_sel+1); b_sel = grid_points(j_sel+1);

            xs_test = rand(test_points,1);
            ys_test = (xs_test >= a_star) & (xs_test <= b_star);
            preds = (xs_test >= a_sel) & (xs_test <= b_sel);

            test_error_frac(t) = mean(preds ~= ys_test);
        end

        mean_test_errors(end+1) = mean(test_error_frac);
    end
end


%% Plot results
figure;
if sweepN
    plot(N_values, mean_test_errors, 'b-o', 'LineWidth', 2); hold on;
    plot(N_values, PAC_required_m, 'r--', 'LineWidth', 2);
    xlabel('N');
    ylabel('Mean test-error(PAC m requirement)');
    legend('Empirical test error', 'PAC required m');
    title('Empirical Test Error vs PAC Bound as N Increases');
else
    plot(m_values, mean_test_errors, 'b-o', 'LineWidth', 2); hold on;
    plot(m_values, PAC_required_m, 'r--', 'LineWidth', 2);
    xlabel('m (training samples)');
    ylabel('Error or PAC m requirement');
    legend('Empirical test error', 'PAC required m');
    title('Empirical Test Error vs PAC Bound as m Increases');
end
grid on;
