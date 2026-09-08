%% ============================================================
% DELTA T - PRESSURE CORRELATION ANALYSIS
%
% Required variables in workspace:
%
%   t       : [T x 1] or [1 x T]
%   p_nodes : [T x N]
%
% For every transition
%
%       X(t_i) -> X(t_{i+1})
%
% we compare
%
%       delta_t(i) = t(i+1) - t(i)
%
% with several measures of pressure magnitude and pressure change.
% ============================================================

clc;


%% ============================================================
% PREPARE DATA
% ============================================================

t = t(:);

if size(p_nodes, 1) ~= numel(t)

    if size(p_nodes, 2) == numel(t)
        p_nodes = p_nodes.';
    else
        error('p_nodes dimensions are incompatible with t.');
    end

end

delta_t = diff(t);

num_steps = numel(delta_t);


%% ============================================================
% PRESSURE AT TARGET TIMESTEP
%
% Sample i predicts:
%
%       p(t_i) -> p(t_{i+1})
%
% Therefore pressure statistics are computed on t_{i+1}.
% ============================================================

p_current = p_nodes(1:end-1, :);
p_target  = p_nodes(2:end, :);


%% ============================================================
% PRESSURE MAGNITUDE STATISTICS
% ============================================================

p_max = max(p_target, [], 2);

p_min = min(p_target, [], 2);

p_mean = mean(p_target, 2);

p_std = std(p_target, 0, 2);

p_range = p_max - p_min;

p_abs_max = max(abs(p_target), [], 2);


%% ============================================================
% PRESSURE CHANGE BETWEEN CONSECUTIVE STATES
%
%       delta_p = p(t_{i+1}) - p(t_i)
% ============================================================

delta_p = p_target - p_current;

% Mean absolute pressure change over the mesh
dp_mae = mean(abs(delta_p), 2);

% RMS pressure change over the mesh
dp_rms = sqrt(mean(delta_p.^2, 2));

% Largest local pressure change
dp_max = max(abs(delta_p), [], 2);


%% ============================================================
% APPROXIMATE TEMPORAL PRESSURE DERIVATIVE
%
%       dp/dt ~= (p(t+dt) - p(t)) / dt
% ============================================================

dp_dt = delta_p ./ delta_t;

% RMS temporal derivative over the mesh
dpdt_rms = sqrt(mean(dp_dt.^2, 2));

% Maximum absolute temporal derivative over the mesh
dpdt_max = max(abs(dp_dt), [], 2);


%% ============================================================
% PEARSON CORRELATIONS
% ============================================================

fprintf('\n');
fprintf('===============================================\n');
fprintf('PEARSON CORRELATION WITH DELTA T\n');
fprintf('===============================================\n');

variables = {
    'p_max',      p_max;
    'p_abs_max',  p_abs_max;
    'p_std',      p_std;
    'p_range',    p_range;
    'dp_mae',     dp_mae;
    'dp_rms',     dp_rms;
    'dp_max',     dp_max;
    'dpdt_rms',   dpdt_rms;
    'dpdt_max',   dpdt_max
};

for i = 1:size(variables, 1)

    name = variables{i, 1};
    values = variables{i, 2};

    R = corrcoef(delta_t, values);

    correlation = R(1, 2);

    fprintf('%-12s : % .6f\n', name, correlation);

end


%% ============================================================
% SPEARMAN CORRELATIONS
%
% Computed manually without Statistics and Machine Learning Toolbox
% ============================================================

fprintf('\n');
fprintf('===============================================\n');
fprintf('SPEARMAN CORRELATION WITH DELTA T\n');
fprintf('===============================================\n');

for i = 1:size(variables, 1)

    name = variables{i, 1};
    values = variables{i, 2};

    % Convert data to ranks
    rank_dt = tied_rank_manual(delta_t);
    rank_values = tied_rank_manual(values);

    % Pearson correlation of the ranks = Spearman correlation
    R = corrcoef(rank_dt, rank_values);

    correlation = R(1, 2);

    fprintf('%-12s : % .6f\n', name, correlation);

end


%% ============================================================
% LOCAL FUNCTION: COMPUTE RANKS
% ============================================================

function ranks = tied_rank_manual(x)

    x = x(:);

    [sorted_x, order] = sort(x);

    ranks = zeros(size(x));

    i = 1;

    while i <= numel(x)

        j = i;

        % Find equal values (ties)
        while j < numel(x) && sorted_x(j + 1) == sorted_x(i)
            j = j + 1;
        end

        % Average rank for tied values
        average_rank = mean(i:j);

        ranks(order(i:j)) = average_rank;

        i = j + 1;

    end

end
%% ============================================================
% SCATTER: DELTA T VS MAX PRESSURE
% ============================================================

figure;

scatter(delta_t, p_max, 20, t(2:end), 'filled');

xlabel('\Delta t');
ylabel('Maximum pressure');

title('\Delta t vs maximum pressure');

cb = colorbar;
cb.Label.String = 'Physical time';

grid on;


%% ============================================================
% SCATTER: DELTA T VS MAX ABSOLUTE PRESSURE
% ============================================================

figure;

scatter(delta_t, p_abs_max, 20, t(2:end), 'filled');

xlabel('\Delta t');
ylabel('Maximum |p|');

title('\Delta t vs maximum absolute pressure');

cb = colorbar;
cb.Label.String = 'Physical time';

grid on;


%% ============================================================
% SCATTER: DELTA T VS RMS PRESSURE CHANGE
% ============================================================

figure;

scatter(delta_t, dp_rms, 20, t(2:end), 'filled');

xlabel('\Delta t');
ylabel('RMS pressure change');

title('\Delta t vs RMS pressure change');

cb = colorbar;
cb.Label.String = 'Physical time';

grid on;


%% ============================================================
% SCATTER: DELTA T VS MAX PRESSURE CHANGE
% ============================================================

figure;

scatter(delta_t, dp_max, 20, t(2:end), 'filled');

xlabel('\Delta t');
ylabel('max |p(t+\Delta t) - p(t)|');

title('\Delta t vs maximum pressure change');

cb = colorbar;
cb.Label.String = 'Physical time';

grid on;


%% ============================================================
% SCATTER: DELTA T VS RMS TEMPORAL PRESSURE DERIVATIVE
% ============================================================

figure;

scatter(delta_t, dpdt_rms, 20, t(2:end), 'filled');

xlabel('\Delta t');
ylabel('RMS |dp/dt|');

title('\Delta t vs RMS temporal pressure derivative');

cb = colorbar;
cb.Label.String = 'Physical time';

grid on;


%% ============================================================
% LOG-LOG: DELTA T VS PRESSURE CHANGE
%
% Particularly useful because delta_t varies strongly during
% the initial transient.
% ============================================================

figure;

loglog(delta_t, dp_rms, 'o');

xlabel('\Delta t');
ylabel('RMS pressure change');

title('\Delta t vs RMS pressure change - log-log');

grid on;


%% ============================================================
% LOG-LOG: DELTA T VS TEMPORAL PRESSURE DERIVATIVE
% ============================================================

figure;

loglog(delta_t, dpdt_rms, 'o');

xlabel('\Delta t');
ylabel('RMS |dp/dt|');

title('\Delta t vs RMS temporal pressure derivative - log-log');

grid on;


%% ============================================================
% TIME EVOLUTION: DELTA T AND PRESSURE
% ============================================================

figure;

yyaxis left

plot(t(2:end), delta_t, 'LineWidth', 1.2);

ylabel('\Delta t');


yyaxis right

plot(t(2:end), p_abs_max, 'LineWidth', 1.2);

ylabel('Maximum |p|');


xlabel('Physical time');

title('\Delta t and maximum pressure vs time');

grid on;


%% ============================================================
% TIME EVOLUTION: DELTA T AND PRESSURE CHANGE
% ============================================================

figure;

yyaxis left

plot(t(2:end), delta_t, 'LineWidth', 1.2);

ylabel('\Delta t');


yyaxis right

plot(t(2:end), dp_rms, 'LineWidth', 1.2);

ylabel('RMS pressure change');


xlabel('Physical time');

title('\Delta t and pressure change vs time');

grid on;


%% ============================================================
% TIME EVOLUTION: DELTA T AND TEMPORAL PRESSURE DERIVATIVE
% ============================================================

figure;

yyaxis left

plot(t(2:end), delta_t, 'LineWidth', 1.2);

ylabel('\Delta t');


yyaxis right

plot(t(2:end), dpdt_rms, 'LineWidth', 1.2);

ylabel('RMS |dp/dt|');


xlabel('Physical time');

title('\Delta t and temporal pressure derivative vs time');

grid on;


%% ============================================================
% STARTUP ZOOM
% ============================================================

target_time = t(2:end);

startup_mask = target_time <= 0.5;


%% ============================================================
% STARTUP: DELTA T AND MAXIMUM PRESSURE
% ============================================================

figure;

yyaxis left

plot( ...
    target_time(startup_mask), ...
    delta_t(startup_mask), ...
    'o-', ...
    'LineWidth', 1.2 ...
);

ylabel('\Delta t');


yyaxis right

plot( ...
    target_time(startup_mask), ...
    p_abs_max(startup_mask), ...
    'o-', ...
    'LineWidth', 1.2 ...
);

ylabel('Maximum |p|');


xlabel('Physical time');

title('Startup: \Delta t and maximum pressure');

grid on;


%% ============================================================
% STARTUP: DELTA T AND PRESSURE VARIATION
% ============================================================

figure;

yyaxis left

plot( ...
    target_time(startup_mask), ...
    delta_t(startup_mask), ...
    'o-', ...
    'LineWidth', 1.2 ...
);

ylabel('\Delta t');


yyaxis right

plot( ...
    target_time(startup_mask), ...
    dp_rms(startup_mask), ...
    'o-', ...
    'LineWidth', 1.2 ...
);

ylabel('RMS pressure change');


xlabel('Physical time');

title('Startup: \Delta t and pressure variation');

grid on;


%% ============================================================
% STARTUP: DELTA T AND TEMPORAL PRESSURE DERIVATIVE
% ============================================================

figure;

yyaxis left

plot( ...
    target_time(startup_mask), ...
    delta_t(startup_mask), ...
    'o-', ...
    'LineWidth', 1.2 ...
);

ylabel('\Delta t');


yyaxis right

plot( ...
    target_time(startup_mask), ...
    dpdt_rms(startup_mask), ...
    'o-', ...
    'LineWidth', 1.2 ...
);

ylabel('RMS |dp/dt|');


xlabel('Physical time');

title('Startup: \Delta t and temporal pressure derivative');

grid on;