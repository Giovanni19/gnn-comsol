%% ============================================================
% GNN -> COMSOL STATIONARY BDF1 - SINGLE TRANSITION
%
% Quick single-transition check: reproduce one transition
%
%       X_k  ->  X_(k+1)
%
% using the stationary BDF1-equivalent problem, comparing the
% STANDARD initial guess (X_k from sol1) against the GNN
% initial guess (X_hat_(k+1) from the trained model).
%
% This is the k_start = k_end special case of
% complete_pipeline_all_timesteps.m - both scripts share the
% same comparison logic, implemented once in
% run_gnn_vs_standard_transitions.m.
% ============================================================

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
% TRANSITION TO TEST
% ============================================================

k = 1000;

if k < 1 || k > num_predictions
    error('Invalid transition index k = %d.', k);
end

fprintf('\n========================================\n');
fprintf('SINGLE STATIONARY BDF1 TEST\n');
fprintf('========================================\n');

fprintf('Transition k = %d\n', k);


%% ============================================================
% RUN THE STANDARD VS GNN COMPARISON
% ============================================================

results = run_gnn_vs_standard_transitions( ...
    model, G, t_comsol, dt_comsol, k);


%% ============================================================
% SPEEDUP FOR THIS TRANSITION
% ============================================================

if results.standard_converged && results.gnn_converged

    speedup = results.standard_time / results.gnn_time;

    fprintf('\n========================================\n');
    fprintf('SPEEDUP\n');
    fprintf('========================================\n');

    fprintf('Standard = %.6f s\n', results.standard_time);
    fprintf('GNN      = %.6f s\n', results.gnn_time);
    fprintf('Speedup  = %.6fx\n', speedup);

end
