function plot_nonlinear_iterations(solver_table)

% ============================================================
% PLOT NONLINEAR ITERATIONS
%
% Compares the number of nonlinear solver iterations required
% by:
%
%   STANDARD initial guess
%   GNN initial guess
%
% as a function of the target physical time.
%
% INPUT
% -----
% solver_table
%     Table produced by run_gnn_vs_standard_transitions.m
%
% ============================================================


%% Extract data

t = solver_table.t_target;

std_iter = solver_table.STD_iterations;
gnn_iter = solver_table.GNN_iterations;


%% Keep only transitions where both methods converged

valid = ...
    solver_table.STD_converged & ...
    solver_table.GNN_converged & ...
    isfinite(std_iter) & ...
    isfinite(gnn_iter);

t_plot = t(valid);

std_plot = std_iter(valid);
gnn_plot = gnn_iter(valid);


%% Create figure

figure;

plot( ...
    t_plot, ...
    std_plot, ...
    '-o', ...
    'LineWidth', 1.5, ...
    'MarkerSize', 4);

hold on;

plot( ...
    t_plot, ...
    gnn_plot, ...
    '-s', ...
    'LineWidth', 1.5, ...
    'MarkerSize', 4);


%% Labels

xlabel('Time [s]');
ylabel('Nonlinear iterations');

title('Nonlinear Solver Iterations: Standard vs GNN');


%% Legend

legend( ...
    'Standard initial guess', ...
    'GNN initial guess', ...
    'Location', 'best');


%% Plot formatting

grid on;
box on;

% Nonlinear iterations are integers
ytickformat('%d');

%% ============================================================
% NONLINEAR ITERATION REDUCTION [%] VS TIME
%
% Positive:
%     GNN requires fewer nonlinear iterations than STANDARD
%
% Zero:
%     Same number of nonlinear iterations
%
% Negative:
%     GNN requires more nonlinear iterations than STANDARD
%
% Definition:
%
% reduction [%] =
%     100 * (STD_iterations - GNN_iterations) / STD_iterations
%
% ============================================================

iteration_reduction_percent = ...
    100 * (std_plot - gnn_plot) ./ std_plot;


%% Create figure

fig_gain = figure;

stem( ...
    t_plot, ...
    iteration_reduction_percent, ...
    'filled', ...
    'LineWidth', 1.3, ...
    'MarkerSize', 4);

hold on;


%% Zero-reference line

yline( ...
    0, ...
    '--', ...
    'LineWidth', 1.2);


%% Labels

xlabel('Time [s]');

ylabel('Nonlinear iteration reduction [%]');

title('GNN Reduction in Nonlinear Solver Iterations');


%% Formatting

grid on;

box on;


%% Save figure

exportgraphics( ...
    fig_gain, ...
    'nonlinear_iteration_reduction_percent_vs_time.png', ...
    'Resolution', 300);

savefig( ...
    fig_gain, ...
    'nonlinear_iteration_reduction_percent_vs_time.fig');


%% Print reduction statistics

fprintf('\n========================================\n');

fprintf('NONLINEAR ITERATION REDUCTION\n');

fprintf('========================================\n');

fprintf('Transitions with GNN improvement = %d\n', ...
    sum(iteration_reduction_percent > 0));

fprintf('Transitions with no change       = %d\n', ...
    sum(iteration_reduction_percent == 0));

fprintf('Transitions where GNN is worse   = %d\n', ...
    sum(iteration_reduction_percent < 0));

fprintf('Mean reduction per timestep      = %.2f %%\n', ...
    mean(iteration_reduction_percent));

fprintf('Median reduction per timestep    = %.2f %%\n', ...
    median(iteration_reduction_percent));
%% Summary

fprintf('\n========================================\n');
fprintf('NONLINEAR ITERATION PLOT\n');
fprintf('========================================\n');

fprintf('Transitions plotted = %d\n', sum(valid));

fprintf('Mean STD iterations = %.3f\n', ...
    mean(std_plot));

fprintf('Mean GNN iterations = %.3f\n', ...
    mean(gnn_plot));

fprintf('Total STD iterations = %d\n', ...
    sum(std_plot));

fprintf('Total GNN iterations = %d\n', ...
    sum(gnn_plot));

fprintf('Total iterations saved = %d\n', ...
    sum(std_plot - gnn_plot));

end