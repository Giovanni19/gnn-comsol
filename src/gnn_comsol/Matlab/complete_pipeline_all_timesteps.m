clear;
clc;
close all;

import com.comsol.model.*
import com.comsol.model.util.*

[model, G, t_comsol, dt_comsol, ...
 num_snapshots, num_transitions, ...
 num_predictions, num_nodes] = ...
    prepare_gnn_comsol_pipeline();


%% ============================================================
% TRANSITION RANGE
% ============================================================

% Test a small range first
k_start = 990;
k_end   = 1010;

if k_start < 1 || k_end > num_predictions || k_start > k_end
    error('Invalid transition range.');
end

k_values = (k_start:k_end).';

fprintf('\n========================================\n');
fprintf('TRANSITION RANGE\n');
fprintf('========================================\n');

fprintf('First transition = %d\n', k_start);
fprintf('Last transition  = %d\n', k_end);
fprintf('Number of tests  = %d\n', length(k_values));


%% ============================================================
% RUN THE STANDARD VS GNN COMPARISON
%
% Shared with complete_pipeline_single_step.m - see
% run_gnn_vs_standard_transitions.m for the solver setup,
% GNN initial-guess handling and per-transition comparison.
% ============================================================

results = run_gnn_vs_standard_transitions( ...
    model, G, t_comsol, dt_comsol, k_values);


%% ============================================================
% SAVE RESULTS
% ============================================================

results_file = 'stationary_gnn_comparison.mat';

save(results_file, 'results');

fprintf('\nResults saved to:\n%s\n', results_file);
