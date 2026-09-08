%% ============================================================
% DELTA T STATISTICS
% ============================================================

clc;

%% ============================================================
% PREPARE TIME VECTOR
% ============================================================

t = t(:);

num_times = numel(t);

if num_times < 2
    error('The time vector must contain at least two values.');
end

delta_t = diff(t);

num_steps = numel(delta_t);

fprintf('\n');
fprintf('========================================\n');
fprintf('TIME VECTOR INFORMATION\n');
fprintf('========================================\n');

fprintf('Number of saved times: %d\n', num_times);
fprintf('Number of delta_t:     %d\n', num_steps);
fprintf('Initial time:          %.8e\n', t(1));
fprintf('Final time:            %.8e\n', t(end));

fprintf('========================================\n');


%% ============================================================
% CONSISTENCY CHECKS
% ============================================================

if any(~isfinite(t))
    error('The time vector contains NaN or Inf values.');
end

if any(delta_t <= 0)
    warning('Some delta_t values are zero or negative.');
end


%% ============================================================
% BASIC STATISTICS
% ============================================================

dt_mean = mean(delta_t);
dt_std = std(delta_t);
dt_median = median(delta_t);

dt_min = min(delta_t);
dt_max = max(delta_t);

dt_cv = dt_std / dt_mean;


%% ============================================================
% MANUAL PERCENTILES
% No Statistics Toolbox required
% ============================================================

dt_sorted = sort(delta_t);

n = numel(dt_sorted);

percentile_value = @(q) interp1( ...
    1:n, ...
    dt_sorted, ...
    1 + (n - 1) * q / 100, ...
    'linear' ...
);

p01  = percentile_value(0.1);
p1   = percentile_value(1);
p5   = percentile_value(5);
p10  = percentile_value(10);
p25  = percentile_value(25);
p50  = percentile_value(50);
p75  = percentile_value(75);
p90  = percentile_value(90);
p95  = percentile_value(95);
p99  = percentile_value(99);
p999 = percentile_value(99.9);


%% ============================================================
% PRINT STATISTICS
% ============================================================

fprintf('\n');
fprintf('========================================\n');
fprintf('DELTA T STATISTICS\n');
fprintf('========================================\n');

fprintf('Mean:        %.8e\n', dt_mean);
fprintf('Std:         %.8e\n', dt_std);
fprintf('Median:      %.8e\n', dt_median);

fprintf('\n');

fprintf('Min:         %.8e\n', dt_min);
fprintf('Max:         %.8e\n', dt_max);

fprintf('\n');

fprintf('P0.1:        %.8e\n', p01);
fprintf('P1:          %.8e\n', p1);
fprintf('P5:          %.8e\n', p5);
fprintf('P10:         %.8e\n', p10);
fprintf('P25:         %.8e\n', p25);
fprintf('P50:         %.8e\n', p50);
fprintf('P75:         %.8e\n', p75);
fprintf('P90:         %.8e\n', p90);
fprintf('P95:         %.8e\n', p95);
fprintf('P99:         %.8e\n', p99);
fprintf('P99.9:       %.8e\n', p999);

fprintf('\n');

fprintf('Coefficient of variation: %.6f\n', dt_cv);

fprintf('========================================\n');


%% ============================================================
% FIRST 50 TIMESTEPS
% ============================================================

num_first = min(50, num_steps);

fprintf('\n');
fprintf('========================================\n');
fprintf('FIRST %d TIME STEPS\n', num_first);
fprintf('========================================\n');

fprintf('%8s %18s %18s %18s\n', ...
    'Step', 't_start', 't_end', 'delta_t');

for i = 1:num_first

    fprintf('%8d %18.8e %18.8e %18.8e\n', ...
        i, t(i), t(i+1), delta_t(i));

end

fprintf('========================================\n');


%% ============================================================
% TIMESTEPS CLOSE TO MAXIMUM DT
% ============================================================

relative_tolerance = 0.01;

close_to_max = abs(delta_t - dt_max) <= ...
    relative_tolerance * abs(dt_max);

num_close_to_max = sum(close_to_max);

fraction_close_to_max = num_close_to_max / num_steps;

fprintf('\n');
fprintf('========================================\n');
fprintf('TIMESTEPS CLOSE TO MAXIMUM DT\n');
fprintf('========================================\n');

fprintf('Maximum dt:               %.8e\n', dt_max);
fprintf('Relative tolerance:       %.2f %%\n', ...
    100 * relative_tolerance);

fprintf('Steps close to max dt:    %d / %d\n', ...
    num_close_to_max, num_steps);

fprintf('Fraction close to max dt: %.2f %%\n', ...
    100 * fraction_close_to_max);

fprintf('========================================\n');


%% ============================================================
% FIRST STEP AFTER WHICH DT REMAINS CLOSE TO MAX
% ============================================================

first_stable_step = NaN;

for i = 1:num_steps

    if all(close_to_max(i:end))
        first_stable_step = i;
        break;
    end

end

fprintf('\n');
fprintf('========================================\n');
fprintf('POSSIBLE TIMESTEP STABILIZATION\n');
fprintf('========================================\n');

if isnan(first_stable_step)

    fprintf(['delta_t never remains within %.2f %% of dt_max ', ...
        'for the rest of the simulation.\n'], ...
        100 * relative_tolerance);

else

    fprintf('First permanently stable step: %d\n', ...
        first_stable_step);

    fprintf('Corresponding start time:      %.8e\n', ...
        t(first_stable_step));

    fprintf('Corresponding delta_t:         %.8e\n', ...
        delta_t(first_stable_step));

end

fprintf('========================================\n');


%% ============================================================
% HISTOGRAM
% ============================================================

figure;

histogram(delta_t, 100);

xlabel('\Delta t');
ylabel('Count');
title('Distribution of \Delta t');

grid on;


%% ============================================================
% HISTOGRAM - LOG COUNTS
% ============================================================

figure;

histogram(delta_t, 100);

set(gca, 'YScale', 'log');

xlabel('\Delta t');
ylabel('Count');
title('Distribution of \Delta t - log count');

grid on;


%% ============================================================
% DELTA T VS STEP INDEX
% ============================================================

figure;

plot(1:num_steps, delta_t, 'LineWidth', 1);

xlabel('Step index');
ylabel('\Delta t');

title('\Delta t over simulation');

grid on;


%% ============================================================
% DELTA T VS PHYSICAL TIME
% ============================================================

figure;

plot(t(1:end-1), delta_t, 'LineWidth', 1);

xlabel('Physical time');
ylabel('\Delta t');

title('\Delta t vs physical time');

grid on;


%% ============================================================
% DELTA T VS PHYSICAL TIME - LOG SCALE
% ============================================================

figure;

semilogy(t(1:end-1), delta_t, 'LineWidth', 1);

xlabel('Physical time');
ylabel('\Delta t');

title('\Delta t vs physical time - log scale');

grid on;


%% ============================================================
% ZOOM ON STARTUP
% ============================================================

num_zoom = min(100, num_steps);

figure;

plot( ...
    t(1:num_zoom), ...
    delta_t(1:num_zoom), ...
    'o-', ...
    'LineWidth', 1 ...
);

xlabel('Physical time');
ylabel('\Delta t');

title(sprintf('\\Delta t during first %d transitions', num_zoom));

grid on;


%% ============================================================
% ZOOM ON STARTUP - LOG SCALE
% ============================================================

figure;

semilogy( ...
    t(1:num_zoom), ...
    delta_t(1:num_zoom), ...
    'o-', ...
    'LineWidth', 1 ...
);

xlabel('Physical time');
ylabel('\Delta t');

title(sprintf( ...
    '\\Delta t during first %d transitions - log scale', ...
    num_zoom ...
));

grid on;


%% ============================================================
% NORMALIZED DELTA T - LOCAL STATISTICS
% ============================================================

delta_t_norm_local = ...
    (delta_t - dt_mean) / (dt_std + 1e-12);

figure;

plot( ...
    t(1:end-1), ...
    delta_t_norm_local, ...
    'LineWidth', 1 ...
);

xlabel('Physical time');
ylabel('Normalized \Delta t');

title('Locally normalized \Delta t vs physical time');

grid on;


%% ============================================================
% NORMALIZED DELTA T - FIRST 100 STEPS
% ============================================================

figure;

plot( ...
    t(1:num_zoom), ...
    delta_t_norm_local(1:num_zoom), ...
    'o-', ...
    'LineWidth', 1 ...
);

xlabel('Physical time');
ylabel('Normalized \Delta t');

title(sprintf( ...
    'Normalized \\Delta t - first %d transitions', ...
    num_zoom ...
));

grid on;


%% ============================================================
% SUCCESSIVE TIMESTEP RATIO
% ============================================================

dt_ratio = delta_t(2:end) ./ delta_t(1:end-1);

figure;

plot( ...
    t(2:end-1), ...
    dt_ratio, ...
    'LineWidth', 1 ...
);

xlabel('Physical time');
ylabel('\Delta t_{i+1} / \Delta t_i');

title('Ratio between consecutive timesteps');

grid on;


%% ============================================================
% RELATIVE CHANGE IN DELTA T
% ============================================================

relative_change = abs( ...
    diff(delta_t) ./ delta_t(1:end-1) ...
);

figure;

semilogy( ...
    t(2:end-1), ...
    relative_change, ...
    'LineWidth', 1 ...
);

xlabel('Physical time');
ylabel('Relative change in \Delta t');

title('Relative change of timestep');

grid on;


%% ============================================================
% PRINT LARGEST CHANGES
% ============================================================

[sorted_change, sorted_indices] = sort( ...
    relative_change, ...
    'descend' ...
);

num_report = min(20, numel(sorted_change));

fprintf('\n');
fprintf('========================================\n');
fprintf('LARGEST RELATIVE CHANGES IN DELTA T\n');
fprintf('========================================\n');

fprintf('%8s %18s %18s %18s %18s\n', ...
    'Step', 'Time', 'dt_old', 'dt_new', 'RelChange');

for k = 1:num_report

    i = sorted_indices(k);

    fprintf('%8d %18.8e %18.8e %18.8e %18.8e\n', ...
        i, ...
        t(i+1), ...
        delta_t(i), ...
        delta_t(i+1), ...
        sorted_change(k));

end

fprintf('========================================\n');