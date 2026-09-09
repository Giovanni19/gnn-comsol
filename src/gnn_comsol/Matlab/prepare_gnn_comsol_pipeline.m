function [model, G, t_comsol, dt_comsol, ...
          num_snapshots, num_transitions, ...
          num_predictions, num_nodes] = ...
          prepare_gnn_comsol_pipeline()

% ============================================================
% PREPARE GNN -> COMSOL PIPELINE
%
% This function performs the common initialization required by
% the GNN-COMSOL stationary experiments:
%
%   1. Load the COMSOL model containing the computed sol1
%   2. Read the time-dependent solution times
%   3. Run Python GNN inference
%   4. Load the GNN predictions
%   5. Check the prediction arrays
%   6. Check COMSOL <-> GNN timestep alignment
%
% Outputs
% -------
% model
%     Loaded COMSOL model.
%
% G
%     Structure containing the GNN predictions.
%
% t_comsol
%     Stored COMSOL solution times.
%
% dt_comsol
%     Time-step duration for each transition.
%
% num_snapshots
%     Number of stored COMSOL snapshots.
%
% num_transitions
%     Number of COMSOL transitions.
%
% num_predictions
%     Number of GNN predictions.
%
% num_nodes
%     Number of GNN nodes.
%
% ============================================================

import com.comsol.model.*
import com.comsol.model.util.*


%% ============================================================
% 1. FILE PATHS
% ============================================================

% All paths are centralized in gnn_comsol_config.m so that every
% pipeline script stays in sync when the model, GNN environment,
% or trained run directory change.
cfg = gnn_comsol_config();

model_file           = cfg.model_file;
dataset_file         = cfg.dataset_file;
python_exe           = cfg.python_exe;
python_script        = cfg.python_script;
run_dir              = cfg.run_dir;
gnn_predictions_file = cfg.gnn_predictions_file;


%% ============================================================
% 2. LOAD COMSOL MODEL
% ============================================================

fprintf('\n========================================\n');
fprintf('LOADING COMSOL MODEL\n');
fprintf('========================================\n');

model = mphload(model_file);

fprintf('COMSOL model loaded successfully.\n');
fprintf('Model:\n%s\n', model_file);


%% ============================================================
% 3. CHECK TIME-DEPENDENT SOLUTION SOL1
% ============================================================

fprintf('\n========================================\n');
fprintf('CHECKING TIME-DEPENDENT SOLUTION\n');
fprintf('========================================\n');

% Read information specifically from sol1
sol1_info = mphsolinfo(model, 'soltag', 'sol1');

% Stored time values
t_comsol = sol1_info.solvals(:);

num_snapshots = length(t_comsol);
num_transitions = num_snapshots - 1;

fprintf('Number of COMSOL snapshots   = %d\n', ...
    num_snapshots);

fprintf('Number of COMSOL transitions = %d\n', ...
    num_transitions);

fprintf('Initial time = %.15e s\n', ...
    t_comsol(1));

fprintf('Final time   = %.15e s\n', ...
    t_comsol(end));

% Time step associated with every transition
dt_comsol = diff(t_comsol);


%% ============================================================
% 4. RUN PYTHON GNN INFERENCE
% ============================================================

fprintf('\n========================================\n');
fprintf('RUNNING GNN INFERENCE\n');
fprintf('========================================\n');

cmd = sprintf( ...
    '"%s" "%s" "%s" --dataset "%s" --no-animation', ...
    python_exe, ...
    python_script, ...
    run_dir, ...
    dataset_file);

fprintf('Python command:\n%s\n\n', cmd);

tic;

[status, python_output] = system(cmd);

gnn_elapsed_time = toc;

% Print Python output in MATLAB
fprintf('%s\n', python_output);

if status ~= 0

    fprintf('\n========================================\n');
    fprintf('GNN INFERENCE FAILED\n');
    fprintf('========================================\n');

    error( ...
        'Python script returned status %d.', ...
        status);

end

fprintf('\n========================================\n');
fprintf('GNN INFERENCE COMPLETED\n');
fprintf('========================================\n');

fprintf('Elapsed time = %.6f s\n', ...
    gnn_elapsed_time);


%% ============================================================
% 5. LOAD GNN PREDICTIONS
% ============================================================

fprintf('\n========================================\n');
fprintf('LOADING GNN PREDICTIONS\n');
fprintf('========================================\n');

if ~isfile(gnn_predictions_file)

    error( ...
        'GNN prediction file not found:\n%s', ...
        gnn_predictions_file);

end

G = load(gnn_predictions_file);

num_predictions = size(G.u_pred, 1);
num_nodes = size(G.u_pred, 2);

fprintf('Number of GNN predictions = %d\n', ...
    num_predictions);

fprintf('Number of GNN nodes       = %d\n', ...
    num_nodes);


%% ============================================================
% 6. BASIC GNN ARRAY CHECKS
% ============================================================

if size(G.v_pred, 1) ~= num_predictions || ...
   size(G.p_pred, 1) ~= num_predictions

    error( ...
        ['u_pred, v_pred and p_pred have inconsistent ' ...
         'numbers of samples.']);

end


if size(G.v_pred, 2) ~= num_nodes || ...
   size(G.p_pred, 2) ~= num_nodes

    error( ...
        ['u_pred, v_pred and p_pred have inconsistent ' ...
         'numbers of nodes.']);

end


dt_gnn = G.delta_t(:);

if length(dt_gnn) ~= num_predictions

    error( ...
        ['Number of GNN delta_t values does not match ' ...
         'predictions.']);

end


%% ============================================================
% 7. CHECK COMSOL <-> GNN ALIGNMENT
% ============================================================

fprintf('\n========================================\n');
fprintf('CHECKING COMSOL <-> GNN ALIGNMENT\n');
fprintf('========================================\n');

% With skip_initial = 0:
%
% GNN prediction 1:
%
%       X_0 -> X_1
%
% GNN prediction 2:
%
%       X_1 -> X_2
%
% Therefore:
%
% number of predictions = number of COMSOL snapshots - 1

if num_predictions ~= num_transitions

    error( ...
        ['COMSOL/GNN transition mismatch.\n' ...
         'COMSOL transitions = %d\n' ...
         'GNN predictions    = %d'], ...
        num_transitions, ...
        num_predictions);

end


% Compare every delta_t
dt_difference = abs(dt_comsol - dt_gnn);

max_dt_error = max(dt_difference);

fprintf( ...
    'Maximum |dt_COMSOL - dt_GNN| = %.15e s\n', ...
    max_dt_error);


% Numerical tolerance
dt_tolerance = ...
    100 * eps(max(abs(dt_comsol)));

if max_dt_error > dt_tolerance

    error( ...
        ['COMSOL and GNN delta_t arrays are not aligned.\n' ...
         'Maximum difference = %.15e s\n' ...
         'Tolerance          = %.15e s'], ...
        max_dt_error, ...
        dt_tolerance);

end

fprintf('\nCOMSOL and GNN are correctly aligned.\n');


%% ============================================================
% 8. SHOW FIRST TRANSITIONS
% ============================================================

fprintf('\n========================================\n');
fprintf('FIRST TRANSITIONS\n');
fprintf('========================================\n');

num_to_print = min(5, num_predictions);

for k = 1:num_to_print

    fprintf( ...
        ['k = %d | ' ...
         't_prev = %.8e | ' ...
         't_target = %.8e | ' ...
         'dt = %.8e\n'], ...
        k, ...
        t_comsol(k), ...
        t_comsol(k+1), ...
        dt_gnn(k));

end


fprintf('\n========================================\n');
fprintf('INITIAL PIPELINE READY\n');
fprintf('========================================\n');

end