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
% 2. SET UP STATIONARY SOLVER
% ============================================================

fprintf('\n========================================\n');
fprintf('SETTING UP STATIONARY SOLVER\n');
fprintf('========================================\n');

% Laminar Flow physics
spf = model.component('comp1').physics('spf');

% Volume Force used for the BDF1-equivalent term
vf = spf.feature('vf1');

% Create the solver sequence only once
model.study('std3').createAutoSequences('sol');

% Dependent Variables node
v1 = model.sol('sol3').feature('v1');

fprintf('Stationary solver ready.\n');

%% ============================================================
% 3. GNN INITIAL-GUESS SETUP
% ============================================================

% GNN node coordinates
x_gnn = G.node_coordinates(:,1);
y_gnn = G.node_coordinates(:,2);

% Coordinates in COMSOL mphinterp format
P_eval = G.node_coordinates.';

% Files read by the COMSOL interpolation functions
u_gnn_file = ...
    'C:\Users\giovanni\.comsol\v64\llmatlab\u_gnn.txt';

v_gnn_file = ...
    'C:\Users\giovanni\.comsol\v64\llmatlab\v_gnn.txt';

p_gnn_file = ...
    'C:\Users\giovanni\.comsol\v64\llmatlab\p_gnn.txt';

%% ============================================================
% 4. TRANSITION RANGE
% ============================================================

% Test a small range first
k_start = 990;
k_end   = 1010;

if k_start < 1 || k_end > num_predictions || k_start > k_end
    error('Invalid transition range.');
end

k_values = (k_start:k_end).';
num_test_steps = length(k_values);

fprintf('\n========================================\n');
fprintf('TRANSITION RANGE\n');
fprintf('========================================\n');

fprintf('First transition = %d\n', k_start);
fprintf('Last transition  = %d\n', k_end);
fprintf('Number of tests  = %d\n', num_test_steps);

%% ============================================================
% 5. PREALLOCATE RESULTS
% ============================================================

standard_time = nan(num_test_steps,1);
gnn_time      = nan(num_test_steps,1);

standard_converged = false(num_test_steps,1);
gnn_converged      = false(num_test_steps,1);

rmse_standard_gnn_u = nan(num_test_steps,1);
rmse_standard_gnn_v = nan(num_test_steps,1);
rmse_standard_gnn_p = nan(num_test_steps,1);

% Stationary Standard vs Time Dependent
rmse_standard_td_u = nan(num_test_steps,1);
rmse_standard_td_v = nan(num_test_steps,1);
rmse_standard_td_p = nan(num_test_steps,1);

% Stationary GNN vs Time Dependent
rmse_gnn_td_u = nan(num_test_steps,1);
rmse_gnn_td_v = nan(num_test_steps,1);
rmse_gnn_td_p = nan(num_test_steps,1);

fprintf('\nAll data structures initialized.\n');

%% ============================================================
% 6. LOOP OVER TRANSITIONS
% ============================================================

fprintf('\n========================================\n');
fprintf('STARTING TRANSITION LOOP\n');
fprintf('========================================\n');

for i = 1:num_test_steps

    % Actual COMSOL / GNN transition index
    k = k_values(i);

    % ---------------------------------------------------------
    % Current transition
    %
    %       X_k -> X_(k+1)
    % ---------------------------------------------------------

    t_prev   = t_comsol(k);
    t_target = t_comsol(k+1);
    dt       = dt_comsol(k);
    %% ========================================================
    % TIME-DEPENDENT TARGET
    %
    % Transition k:
    %
    %       X_k -> X_(k+1)
    %
    % The reference target is therefore sol1(k+1).
    % =========================================================

    u_td = mphinterp(model, 'u', ...
        'coord', P_eval, ...
        'dataset', 'dset1', ...
        'solnum', k+1);

    v_td = mphinterp(model, 'v', ...
        'coord', P_eval, ...
        'dataset', 'dset1', ...
        'solnum', k+1);

    p_td = mphinterp(model, 'p', ...
        'coord', P_eval, ...
        'dataset', 'dset1', ...
        'solnum', k+1);

    u_td = u_td(:);
    v_td = v_td(:);
    p_td = p_td(:);

    fprintf('\n----------------------------------------\n');
    fprintf('Transition %d/%d | k = %d\n', ...
        i, num_test_steps, k);
    fprintf('t_prev   = %.15e s\n', t_prev);
    fprintf('t_target = %.15e s\n', t_target);
    fprintf('dt       = %.15e s\n', dt);


    %% ========================================================
    % 6.1 UPDATE TIME STEP
    % =========================================================

    dt_str = sprintf('%.17g[s]', dt);

    model.param.set('dt_step', dt_str);


    %% ========================================================
    % 6.2 UPDATE BDF1-EQUIVALENT VOLUME FORCE
    %
    % For this transition:
    %
    %   Fx = rho*(u_k - u)/dt_k
    %   Fy = rho*(v_k - v)/dt_k
    %
    % u_k and v_k are read directly from sol1.
    % =========================================================

    u_prev_expr = sprintf( ...
        "withsol('sol1',u,setind(t,%d))", ...
        k);

    v_prev_expr = sprintf( ...
        "withsol('sol1',v,setind(t,%d))", ...
        k);

    Fx = sprintf( ...
        'spf.rho*(%s-u)/dt_step', ...
        u_prev_expr);

    Fy = sprintf( ...
        'spf.rho*(%s-v)/dt_step', ...
        v_prev_expr);

    vf.set('F', {Fx; Fy; '0'});


    fprintf('Stationary problem updated.\n');

    %% ========================================================
    % 6.3 CONFIGURE STANDARD INITIAL GUESS
    %
    % Standard nonlinear initial guess:
    %
    %       X^(0) = X_k
    %
    % taken directly from the time-dependent solution sol1.
    % =========================================================

    v1.set('initmethod', 'sol');
    v1.set('initsol', 'sol1');
    v1.set('solnum', num2str(k));


    %% ========================================================
    % 6.4 RUN STANDARD STATIONARY SOLVER
    % =========================================================

    fprintf('Running standard stationary solver... ');

    tic;

    try

        model.sol('sol3').runAll();

        standard_time(i) = toc;
        standard_converged(i) = true;

        fprintf('CONVERGED | %.6f s\n', ...
            standard_time(i));

    catch ME

        standard_time(i) = toc;
        standard_converged(i) = false;

        fprintf('FAILED | %.6f s\n', ...
            standard_time(i));

        fprintf('Error at k = %d:\n%s\n', ...
            k, ME.message);

    end
        %% ========================================================
    % 6.5 STORE STANDARD SOLUTION
    % =========================================================

    if standard_converged(i)

        u_standard = mphinterp(model, 'u', ...
            'coord', P_eval, ...
            'dataset', 'dset3');

        v_standard = mphinterp(model, 'v', ...
            'coord', P_eval, ...
            'dataset', 'dset3');

        p_standard = mphinterp(model, 'p', ...
            'coord', P_eval, ...
            'dataset', 'dset3');

        u_standard = u_standard(:);
        v_standard = v_standard(:);
        p_standard = p_standard(:);
        % ----------------------------------------------------
        % Standard Stationary vs Time Dependent
        % ----------------------------------------------------

        rmse_standard_td_u(i) = sqrt( ...
            mean((u_standard - u_td).^2));

        rmse_standard_td_v(i) = sqrt( ...
            mean((v_standard - v_td).^2));

        rmse_standard_td_p(i) = sqrt( ...
            mean((p_standard - p_td).^2));

    end


    %% ========================================================
    % 6.6 GET GNN PREDICTION
    %
    % Prediction of X_(k+1)
    % =========================================================

    u_gnn_pred = G.u_pred(k,:).';
    v_gnn_pred = G.v_pred(k,:).';
    p_gnn_pred = G.p_pred(k,:).';


    %% ========================================================
    % 6.7 UPDATE GNN INTERPOLATION FILES
    % =========================================================

    writematrix( ...
        [x_gnn, y_gnn, u_gnn_pred], ...
        u_gnn_file, ...
        'Delimiter', 'space');

    writematrix( ...
        [x_gnn, y_gnn, v_gnn_pred], ...
        v_gnn_file, ...
        'Delimiter', 'space');

    writematrix( ...
        [x_gnn, y_gnn, p_gnn_pred], ...
        p_gnn_file, ...
        'Delimiter', 'space');

    % Make COMSOL reload the new prediction
    model.func('int4').refresh();
    model.func('int5').refresh();
    model.func('int6').refresh();


    %% ========================================================
    % 6.8 CONFIGURE GNN INITIAL GUESS
    %
    % The stationary equations and F are unchanged.
    %
    % Only the nonlinear initial guess changes:
    %
    %       X^(0) = X_hat_(k+1)^GNN
    % =========================================================

    v1.set('initmethod', 'init');


    %% ========================================================
    % 6.9 RUN GNN-INITIALIZED STATIONARY SOLVER
    % =========================================================

    fprintf('Running GNN stationary solver... ');

    tic;

    try

        model.sol('sol3').runAll();

        gnn_time(i) = toc;
        gnn_converged(i) = true;

        fprintf('CONVERGED | %.6f s\n', ...
            gnn_time(i));

    catch ME

        gnn_time(i) = toc;
        gnn_converged(i) = false;

        fprintf('FAILED | %.6f s\n', ...
            gnn_time(i));

        fprintf('GNN error at k = %d:\n%s\n', ...
            k, ME.message);

    end


    %% ========================================================
    % 6.10 STORE GNN SOLUTION
    % =========================================================

    if gnn_converged(i)

        u_gnn = mphinterp(model, 'u', ...
            'coord', P_eval, ...
            'dataset', 'dset3');

        v_gnn = mphinterp(model, 'v', ...
            'coord', P_eval, ...
            'dataset', 'dset3');

        p_gnn = mphinterp(model, 'p', ...
            'coord', P_eval, ...
            'dataset', 'dset3');

        u_gnn = u_gnn(:);
        v_gnn = v_gnn(:);
        p_gnn = p_gnn(:);
        % ----------------------------------------------------
        % GNN Stationary vs Time Dependent
        % ----------------------------------------------------

        rmse_gnn_td_u(i) = sqrt( ...
            mean((u_gnn - u_td).^2));

        rmse_gnn_td_v(i) = sqrt( ...
            mean((v_gnn - v_td).^2));

        rmse_gnn_td_p(i) = sqrt( ...
            mean((p_gnn - p_td).^2));

    end


    %% ========================================================
    % 6.11 COMPARE FINAL STATIONARY SOLUTIONS
    % =========================================================

    if standard_converged(i) && gnn_converged(i)

        rmse_standard_gnn_u(i) = sqrt( ...
            mean((u_gnn - u_standard).^2));

        rmse_standard_gnn_v(i) = sqrt( ...
            mean((v_gnn - v_standard).^2));

        rmse_standard_gnn_p(i) = sqrt( ...
            mean((p_gnn - p_standard).^2));

    end


    %% ========================================================
    % 6.12 TRANSITION SUMMARY
    % =========================================================

        fprintf( ...
        ['k=%d | STD=%d (%.3fs) | GNN=%d (%.3fs)\n' ...
         '       STATstd-GNN: RMSE u=%.3e\n' ...
         '       STATstd-TD : RMSE u=%.3e | v=%.3e | p=%.3e\n'], ...
        k, ...
        standard_converged(i), ...
        standard_time(i), ...
        gnn_converged(i), ...
        gnn_time(i), ...
        rmse_standard_gnn_u(i), ...
        rmse_standard_td_u(i), ...
        rmse_standard_td_v(i), ...
        rmse_standard_td_p(i));

    
end

%% ============================================================
% 7. FINAL SUMMARY
% ============================================================

fprintf('\n========================================\n');
fprintf('FINAL SUMMARY\n');
fprintf('========================================\n');

fprintf('Transitions tested: %d\n\n', ...
    num_test_steps);

fprintf('Standard converged: %d / %d\n', ...
    sum(standard_converged), ...
    num_test_steps);

fprintf('GNN converged:      %d / %d\n', ...
    sum(gnn_converged), ...
    num_test_steps);


% ------------------------------------------------------------
% Mean solve times
% ------------------------------------------------------------

if any(standard_converged)

    mean_standard_time = mean( ...
        standard_time(standard_converged));

else

    mean_standard_time = NaN;

end


if any(gnn_converged)

    mean_gnn_time = mean( ...
        gnn_time(gnn_converged));

else

    mean_gnn_time = NaN;

end


fprintf('\nMean standard time = %.6f s\n', ...
    mean_standard_time);

fprintf('Mean GNN time      = %.6f s\n', ...
    mean_gnn_time);


% ------------------------------------------------------------
% Compare timing only where BOTH solves converged
% ------------------------------------------------------------

both_converged = ...
    standard_converged & gnn_converged;

if any(both_converged)

    paired_standard_time = ...
        standard_time(both_converged);

    paired_gnn_time = ...
        gnn_time(both_converged);

    total_standard_time = ...
        sum(paired_standard_time);

    total_gnn_time = ...
        sum(paired_gnn_time);

    total_speedup = ...
        total_standard_time / total_gnn_time;

    fprintf('\nBoth converged: %d / %d\n', ...
        sum(both_converged), ...
        num_test_steps);

    fprintf('Paired standard total = %.6f s\n', ...
        total_standard_time);

    fprintf('Paired GNN total      = %.6f s\n', ...
        total_gnn_time);

    fprintf('Paired speedup        = %.6fx\n', ...
        total_speedup);

else

    fprintf('\nNo transitions converged with both methods.\n');

end

%% ============================================================
% 8. SAVE RESULTS
% ============================================================

results.k = k_values;

results.t_prev = t_comsol(k_values);
results.t_target = t_comsol(k_values + 1);
results.dt = dt_comsol(k_values);

results.standard_time = standard_time;
results.gnn_time = gnn_time;

results.standard_converged = standard_converged;
results.gnn_converged = gnn_converged;

results.rmse_standard_gnn_u = rmse_standard_gnn_u;
results.rmse_standard_gnn_v = rmse_standard_gnn_v;
results.rmse_standard_gnn_p = rmse_standard_gnn_p;

results.rmse_standard_td_u = rmse_standard_td_u;
results.rmse_standard_td_v = rmse_standard_td_v;
results.rmse_standard_td_p = rmse_standard_td_p;

results.rmse_gnn_td_u = rmse_gnn_td_u;
results.rmse_gnn_td_v = rmse_gnn_td_v;
results.rmse_gnn_td_p = rmse_gnn_td_p;

results_file = ...
    'stationary_gnn_comparison.mat';

save(results_file, 'results');

fprintf('\nResults saved to:\n%s\n', results_file);

