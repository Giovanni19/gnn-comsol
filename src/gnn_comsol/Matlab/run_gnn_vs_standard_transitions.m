function results = run_gnn_vs_standard_transitions( ...
    model, G, t_comsol, dt_comsol, k_values, bdf_order)

% ============================================================
% RUN GNN VS STANDARD TRANSITIONS
%
% Shared core of the GNN -> COMSOL stationary BDF1 comparison.
%
% For every transition k in k_values:
%
%       X_k -> X_(k+1)
%
% solves the stationary BDF1-equivalent problem twice:
%
%   1. STANDARD initial guess: X^(0) = X_k        (from sol1)
%   2. GNN initial guess:      X^(0) = X_hat_(k+1) (from the GNN)
%
% and compares both against each other and against the
% time-dependent target X_(k+1) stored in sol1.
%
% Used by both complete_pipeline_single_step.m (a single k) and
% complete_pipeline_all_timesteps.m (a range of k values), so the
% comparison logic only needs to be maintained in one place.
%
% Inputs
% ------
% model, G, t_comsol, dt_comsol
%     Outputs of prepare_gnn_comsol_pipeline().
%
% k_values
%     Column (or row) vector of transition indices to test.
%
% Output
% ------
% results
%     Struct with one row per transition in k_values:
%     k, t_prev, t_target, dt,
%     standard_time, gnn_time, standard_converged, gnn_converged,
%     rmse_standard_gnn_{u,v,p}, rmse_standard_td_{u,v,p},
%     rmse_gnn_td_{u,v,p}.
% ============================================================
import com.comsol.model.util.*
k_values = k_values(:);
num_test_steps = length(k_values);

% Validate BDF order
if ~ismember(bdf_order, [1, 2])
    error('bdf_order must be either 1 or 2.');
end

% BDF2 needs X_(k-1), X_k and solves for X_(k+1)
if bdf_order == 2 && any(k_values < 2)
    error( ...
        ['BDF2 requires two previous states. ' ...
         'All transition indices must satisfy k >= 2.']);
end

fprintf('\nTime discretization: BDF%d\n', bdf_order);

%% ============================================================
% 1. SET UP STATIONARY SOLVER
% ============================================================

fprintf('\n========================================\n');
fprintf('SETTING UP STATIONARY SOLVER\n');
fprintf('========================================\n');

% Laminar Flow physics
spf = model.component('comp1').physics('spf');

% Physics initial values
init1 = spf.feature('init1');

% Volume Force used for the BDF1-equivalent term
vf = spf.feature('vf1');

% Create the solver sequence only once
%model.study('std3').createAutoSequences('sol');

% Dependent Variables node
v1 = model.sol('sol3').feature('v1');

fprintf('Stationary solver ready.\n');


%% ============================================================
% 2. GNN INITIAL-GUESS SETUP
% ============================================================

% GNN node coordinates
x_gnn = G.node_coordinates(:,1);
y_gnn = G.node_coordinates(:,2);

% Coordinates in COMSOL mphinterp format
P_eval = G.node_coordinates.';

% Files read by the COMSOL interpolation functions
cfg = gnn_comsol_config();

u_gnn_file = cfg.u_gnn_file;
v_gnn_file = cfg.v_gnn_file;
p_gnn_file = cfg.p_gnn_file;


%% ============================================================
% 3. PREALLOCATE RESULTS
% ============================================================

standard_time = nan(num_test_steps,1);
gnn_time      = nan(num_test_steps,1);

standard_iterations = nan(num_test_steps,1);
gnn_iterations      = nan(num_test_steps,1);

% Cumulative nonlinear solver statistics
standard_res_evals = nan(num_test_steps,1);
gnn_res_evals      = nan(num_test_steps,1);

standard_jac_evals = nan(num_test_steps,1);
gnn_jac_evals      = nan(num_test_steps,1);

standard_linear_solves = nan(num_test_steps,1);
gnn_linear_solves      = nan(num_test_steps,1);

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

% Spatial error fields
error_standard_td_u = cell(num_test_steps,1);
error_standard_td_v = cell(num_test_steps,1);
error_standard_td_p = cell(num_test_steps,1);

error_gnn_td_u = cell(num_test_steps,1);
error_gnn_td_v = cell(num_test_steps,1);
error_gnn_td_p = cell(num_test_steps,1);

% Time-discretization information
dt_previous = nan(num_test_steps,1);
dt_current  = nan(num_test_steps,1);
step_ratio  = nan(num_test_steps,1);

bdf_a0 = nan(num_test_steps,1);
bdf_a1 = nan(num_test_steps,1);
bdf_a2 = nan(num_test_steps,1);

fprintf('\nAll data structures initialized.\n');


%% ============================================================
% 4. LOOP OVER TRANSITIONS
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
    
    % Actual current time-step
    dt = t_target - t_prev;
    
    dt_current(i) = dt;

    if bdf_order == 2

        t_prevprev = t_comsol(k-1);
    
        dt_previous(i) = ...
            t_prev - t_prevprev;
    
        step_ratio(i) = ...
            dt_current(i) / dt_previous(i);
    
    else
    
        t_prevprev = NaN;
    
    end
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
        'dataset', 'dset4', ...
        'solnum', k+1);

    v_td = mphinterp(model, 'v', ...
        'coord', P_eval, ...
        'dataset', 'dset4', ...
        'solnum', k+1);

    p_td = mphinterp(model, 'p', ...
        'coord', P_eval, ...
        'dataset', 'dset4', ...
        'solnum', k+1);

    u_td = u_td(:);
    v_td = v_td(:);
    p_td = p_td(:);

    fprintf('\n----------------------------------------\n');
    fprintf('Transition %d/%d | k = %d\n', ...
        i, num_test_steps, k);

    %% ========================================================
    % 4.1 UPDATE TIME STEP
    % =========================================================

    dt_str = sprintf('%.17g[s]', dt);

    model.param.set('dt_step', dt_str);


    %% ========================================================
    % 4.2 UPDATE BDF-EQUIVALENT VOLUME FORCE
    %
    % BDF1:
    %
    %   (u_(k+1) - u_k) / h_k
    %
    % BDF2 variable-step:
    %
    %   [a0*u_(k+1) + a1*u_k + a2*u_(k-1)] / h_k
    %
    % where
    %
    %   w  = h_k / h_(k-1)
    %
    %   a0 = (1 + 2*w)/(1 + w)
    %   a1 = -(1 + w)
    %   a2 = w^2/(1 + w)
    %
    % The time-derivative term is introduced as a volume force
    % with opposite sign.
    % =========================================================
    
    % Current state X_k
    u_prev_expr = sprintf( ...
        "withsol('sol4',u,setind(t,%d))", ...
        k);
    
    v_prev_expr = sprintf( ...
        "withsol('sol4',v,setind(t,%d))", ...
        k);
    
    
    if bdf_order == 1
    
        % =====================================================
        % BDF1
        %
        % du/dt = (u_(k+1) - u_k)/h_k
        %
        % Fx = rho*(u_k - u)/h_k
        % =====================================================
    
        bdf_a0(i) = 1;
        bdf_a1(i) = -1;
        bdf_a2(i) = 0;
    
        Fx = sprintf( ...
            'spf.rho*(%s-u)/dt_step', ...
            u_prev_expr);
    
        Fy = sprintf( ...
            'spf.rho*(%s-v)/dt_step', ...
            v_prev_expr);
    
    
    elseif bdf_order == 2
    
        % =====================================================
        % VARIABLE-STEP BDF2
        % =====================================================
    
        h_n   = dt_current(i);
        h_nm1 = dt_previous(i);
    
        if h_n <= 0 || h_nm1 <= 0
            error( ...
                'Non-positive timestep detected at k=%d.', ...
                k);
        end
    
        w_n = h_n / h_nm1;
    
        % Variable-step BDF2 coefficients
        a0 = (1 + 2*w_n) / (1 + w_n);
        a1 = -(1 + w_n);
        a2 = w_n^2 / (1 + w_n);
    
        bdf_a0(i) = a0;
        bdf_a1(i) = a1;
        bdf_a2(i) = a2;
    
        % Previous-previous state X_(k-1)
        u_prevprev_expr = sprintf( ...
            "withsol('sol4',u,setind(t,%d))", ...
            k-1);
    
        v_prevprev_expr = sprintf( ...
            "withsol('sol4',v,setind(t,%d))", ...
            k-1);
    
        % -----------------------------------------------------
        % Equivalent stationary force:
        %
        % Fx = -rho/h_n *
        %      (a0*u + a1*u_k + a2*u_(k-1))
        %
        % Fy = -rho/h_n *
        %      (a0*v + a1*v_k + a2*v_(k-1))
        % -----------------------------------------------------
    
        Fx = sprintf( ...
            ['-spf.rho*(%.17g*u + %.17g*(%s) ' ...
             '+ %.17g*(%s))/dt_step'], ...
            a0, ...
            a1, u_prev_expr, ...
            a2, u_prevprev_expr);
    
        Fy = sprintf( ...
            ['-spf.rho*(%.17g*v + %.17g*(%s) ' ...
             '+ %.17g*(%s))/dt_step'], ...
            a0, ...
            a1, v_prev_expr, ...
            a2, v_prevprev_expr);
    
    
    end
    
    
    % Apply volume force
    vf.set('F', {Fx; Fy; '0'});
    
    fprintf( ...
        'Stationary BDF%d problem updated.\n', ...
        bdf_order);
    %% ========================================================
    % 4.3 CONFIGURE STANDARD INITIAL GUESS
    %
    % Standard nonlinear initial guess:
    %
    %       X^(0) = X_k
    %
    % taken directly from the time-dependent solution sol1.
    % =========================================================
    % Reset physics initial values.
    % STANDARD does not use them because initmethod='sol',
    % but this prevents the GNN configuration from the previous
    % transition from remaining active in the model.
    init1.set('u_init', {'0'; '0'; '0'});
    init1.set('p_init', '0');

    % STANDARD initial guess = previous TD solution
    v1.set('initmethod', 'sol');
    v1.set('initsol', 'sol4');
    v1.set('solnum', num2str(k));


    %% ========================================================
    % 4.4 RUN STANDARD STATIONARY SOLVER
    % =========================================================
    
    fprintf('Running standard stationary solver... ');
    
    standard_log = fullfile( ...
        tempdir, ...
        sprintf('comsol_standard_k%d.log', k));
    
    if isfile(standard_log)
        delete(standard_log);
    end
    
    ModelUtil.showProgress(standard_log);
    
    tic;
    
    try
    
        model.sol('sol3').runAll();
    
        standard_time(i) = toc;
        standard_converged(i) = true;
    
        ModelUtil.showProgress(false);
    
        [standard_iterations(i), standard_info] = ...
            get_nonlinear_iterations(standard_log);

        standard_res_evals(i) = ...
            standard_info.total_residual_evaluations;
    
        standard_jac_evals(i) = ...
            standard_info.total_jacobian_evaluations;
        
        standard_linear_solves(i) = ...
            standard_info.total_linear_solves;
        
        fprintf( ...
            ['CONVERGED | %.6f s | Iter=%d | ' ...
             '#Res=%d | #Jac=%d | #Sol=%d\n'], ...
            standard_time(i), ...
            standard_iterations(i), ...
            standard_res_evals(i), ...
            standard_jac_evals(i), ...
            standard_linear_solves(i));

        
    
    catch ME
    
        standard_time(i) = toc;
        standard_converged(i) = false;
    
        ModelUtil.showProgress(false);
    
        fprintf('FAILED | %.6f s\n', ...
            standard_time(i));
    
        fprintf('Error at k = %d:\n%s\n', ...
            k, ME.message);
    
    end

    %% ========================================================
    % 4.5 STORE STANDARD SOLUTION
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
        % ----------------------------------------------------
        % Spatial error: Standard Stationary vs Time Dependent
        % ----------------------------------------------------
        
        error_standard_td_u{i} = abs(u_standard - u_td);
        error_standard_td_v{i} = abs(v_standard - v_td);
        error_standard_td_p{i} = abs(p_standard - p_td);
        
        rmse_standard_td_u(i) = sqrt( ...
            mean((u_standard - u_td).^2));
        
        rmse_standard_td_v(i) = sqrt( ...
            mean((v_standard - v_td).^2));
        
        rmse_standard_td_p(i) = sqrt( ...
            mean((p_standard - p_td).^2));

    end


    %% ========================================================
    % 4.6 GET GNN PREDICTION
    %
    % Prediction of X_(k+1)
    % =========================================================

    u_gnn_pred = G.u_pred(k,:).';
    v_gnn_pred = G.v_pred(k,:).';
    p_gnn_pred = G.p_pred(k,:).';


    %% ========================================================
    % 4.7 UPDATE GNN INTERPOLATION FILES
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
    
    
    % int4/int5/int6 are the interpolation FEATURE TAGS.
    % The callable COMSOL functions are uGNN/vGNN/pGNN.
    init1.set('u_init', {'uGNN(x,y)'; 'vGNN(x,y)'; '0'});
    init1.set('p_init', 'pGNN(x,y)');
    
    % Use physics initial values as nonlinear initial guess
    % 4.8 CONFIGURE GNN INITIAL GUESS
    %
    % The stationary equations and F are unchanged.
    %
    % Only the nonlinear initial guess changes:
    %
    %       X^(0) = X_hat_(k+1)^GNN
    % =========================================================

    v1.set('initmethod', 'init');


   %% ========================================================
    % 4.9 RUN GNN-INITIALIZED STATIONARY SOLVER
    % =========================================================
    fprintf('\n========================================\n');
    fprintf('GNN INITIAL GUESS CHECK\n');
    fprintf('========================================\n');
    
    u_init_check = init1.getStringArray('u_init');
    p_init_check = init1.getStringArray('p_init');
    
    disp(u_init_check);
    disp(p_init_check);
    
    fprintf('initmethod = %s\n', ...
        char(v1.getString('initmethod')));
    fprintf('Running GNN stationary solver... ');
    
    gnn_log = fullfile( ...
        tempdir, ...
        sprintf('comsol_gnn_k%d.log', k));
    
    if isfile(gnn_log)
        delete(gnn_log);
    end
    
    ModelUtil.showProgress(gnn_log);
    
    tic;
    
    try
    
        model.sol('sol3').runAll();
    
        gnn_time(i) = toc;
        gnn_converged(i) = true;
    
        ModelUtil.showProgress(false);
    
        [gnn_iterations(i), gnn_info] = ...
            get_nonlinear_iterations(gnn_log);
        gnn_res_evals(i) = ...
            gnn_info.total_residual_evaluations;
        
        gnn_jac_evals(i) = ...
            gnn_info.total_jacobian_evaluations;
        
        gnn_linear_solves(i) = ...
            gnn_info.total_linear_solves;
    
        fprintf( ...
            ['CONVERGED | %.6f s | Iter=%d | ' ...
             '#Res=%d | #Jac=%d | #Sol=%d\n'], ...
            gnn_time(i), ...
            gnn_iterations(i), ...
            gnn_res_evals(i), ...
            gnn_jac_evals(i), ...
            gnn_linear_solves(i));
        
    
    catch ME
    
        gnn_time(i) = toc;
        gnn_converged(i) = false;
    
        ModelUtil.showProgress(false);
    
        fprintf('FAILED | %.6f s\n', ...
            gnn_time(i));
    
        fprintf('GNN error at k = %d:\n%s\n', ...
            k, ME.message);
    
    end

    %% ========================================================
    % 4.10 STORE GNN SOLUTION
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
        error_gnn_td_u{i} = abs(u_gnn - u_td);
        error_gnn_td_v{i} = abs(v_gnn - v_td);
        error_gnn_td_p{i} = abs(p_gnn - p_td);
        rmse_gnn_td_u(i) = sqrt( ...
            mean((u_gnn - u_td).^2));

        rmse_gnn_td_v(i) = sqrt( ...
            mean((v_gnn - v_td).^2));

        rmse_gnn_td_p(i) = sqrt( ...
            mean((p_gnn - p_td).^2));

    end


    %% ========================================================
    % 4.11 COMPARE FINAL STATIONARY SOLUTIONS
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
    % 4.12 TRANSITION SUMMARY
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
    
    if standard_converged(i) && gnn_converged(i)

        iterations_saved = ...
            standard_iterations(i) - gnn_iterations(i);
    
        iteration_reduction = ...
            100 * iterations_saved / standard_iterations(i);
    
        fprintf( ...
            ['       Newton iterations: STD=%d | GNN=%d | ' ...
             'saved=%d | reduction=%.2f%%\n'], ...
            standard_iterations(i), ...
            gnn_iterations(i), ...
            iterations_saved, ...
            iteration_reduction);
        fprintf( ...
            ['       Solver work:       ' ...
             '#Res STD=%d GNN=%d | ' ...
             '#Jac STD=%d GNN=%d | ' ...
             '#Sol STD=%d GNN=%d\n'], ...
            standard_res_evals(i), ...
            gnn_res_evals(i), ...
            standard_jac_evals(i), ...
            gnn_jac_evals(i), ...
            standard_linear_solves(i), ...
            gnn_linear_solves(i));

    end

end


%% ============================================================
% 5. FINAL SUMMARY
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
    mean_standard_time = mean(standard_time(standard_converged));
else
    mean_standard_time = NaN;
end

if any(gnn_converged)
    mean_gnn_time = mean(gnn_time(gnn_converged));
else
    mean_gnn_time = NaN;
end

fprintf('\nMean standard time = %.6f s\n', mean_standard_time);
fprintf('Mean GNN time      = %.6f s\n', mean_gnn_time);


% ------------------------------------------------------------
% Compare timing only where BOTH solves converged
% ------------------------------------------------------------

both_converged = standard_converged & gnn_converged;

if any(both_converged)

    paired_standard_time = standard_time(both_converged);
    paired_gnn_time      = gnn_time(both_converged);

    total_standard_time = sum(paired_standard_time);
    total_gnn_time      = sum(paired_gnn_time);

    total_speedup = total_standard_time / total_gnn_time;

    fprintf('\nBoth converged: %d / %d\n', ...
        sum(both_converged), num_test_steps);

    fprintf('Paired standard total = %.6f s\n', total_standard_time);
    fprintf('Paired GNN total      = %.6f s\n', total_gnn_time);
    fprintf('Paired speedup        = %.6fx\n', total_speedup);

else

    fprintf('\nNo transitions converged with both methods.\n');

end


%% ============================================================
% 6. BUILD RESULTS STRUCT
% ============================================================

results.k = k_values;

results.t_prev   = t_comsol(k_values);
results.t_target = t_comsol(k_values + 1);
results.dt = dt_current;

results.standard_time = standard_time;
results.gnn_time      = gnn_time;
results.standard_iterations = standard_iterations;
results.gnn_iterations      = gnn_iterations;

% Nonlinear solver work

results.standard_res_evals = standard_res_evals;
results.gnn_res_evals      = gnn_res_evals;

results.standard_jac_evals = standard_jac_evals;
results.gnn_jac_evals      = gnn_jac_evals;

results.standard_linear_solves = standard_linear_solves;
results.gnn_linear_solves      = gnn_linear_solves;

results.iterations_saved = ...
    standard_iterations - gnn_iterations;

results.iteration_reduction_percent = ...
    100 * ...
    (standard_iterations - gnn_iterations) ...
    ./ standard_iterations;


results.standard_converged = standard_converged;
results.gnn_converged      = gnn_converged;

results.rmse_standard_gnn_u = rmse_standard_gnn_u;
results.rmse_standard_gnn_v = rmse_standard_gnn_v;
results.rmse_standard_gnn_p = rmse_standard_gnn_p;

results.rmse_standard_td_u = rmse_standard_td_u;
results.rmse_standard_td_v = rmse_standard_td_v;
results.rmse_standard_td_p = rmse_standard_td_p;

results.rmse_gnn_td_u = rmse_gnn_td_u;
results.rmse_gnn_td_v = rmse_gnn_td_v;
results.rmse_gnn_td_p = rmse_gnn_td_p;

results.node_coordinates = G.node_coordinates;

results.error_standard_td_u = error_standard_td_u;
results.error_standard_td_v = error_standard_td_v;
results.error_standard_td_p = error_standard_td_p;

results.error_gnn_td_u = error_gnn_td_u;
results.error_gnn_td_v = error_gnn_td_v;
results.error_gnn_td_p = error_gnn_td_p;

results.bdf_order = bdf_order;

results.dt_current = dt_current;
results.dt_previous = dt_previous;
results.step_ratio = step_ratio;

results.bdf_a0 = bdf_a0;
results.bdf_a1 = bdf_a1;
results.bdf_a2 = bdf_a2;

%% ============================================================
% 7. BUILD SOLVER COMPARISON TABLE
% ============================================================

solver_table = table( ...
    results.k, ...
    results.t_prev, ...
    results.t_target, ...
    results.dt, ...
    results.standard_converged, ...
    results.gnn_converged, ...
    results.standard_iterations, ...
    results.gnn_iterations, ...
    results.standard_res_evals, ...
    results.gnn_res_evals, ...
    results.standard_jac_evals, ...
    results.gnn_jac_evals, ...
    results.standard_linear_solves, ...
    results.gnn_linear_solves, ...
    results.standard_time, ...
    results.gnn_time, ...
    'VariableNames', { ...
        'k', ...
        't_prev', ...
        't_target', ...
        'dt', ...
        'STD_converged', ...
        'GNN_converged', ...
        'STD_iterations', ...
        'GNN_iterations', ...
        'STD_res_evals', ...
        'GNN_res_evals', ...
        'STD_jac_evals', ...
        'GNN_jac_evals', ...
        'STD_linear_solves', ...
        'GNN_linear_solves', ...
        'STD_time', ...
        'GNN_time' ...
    });

solver_table.BDF_order = ...
    repmat(bdf_order, height(solver_table), 1);

solver_table.dt_previous = dt_previous;
solver_table.dt_current  = dt_current;
solver_table.step_ratio  = step_ratio;

solver_table.BDF_a0 = bdf_a0;
solver_table.BDF_a1 = bdf_a1;
solver_table.BDF_a2 = bdf_a2;

solver_table.iterations_saved = ...
    solver_table.STD_iterations - solver_table.GNN_iterations;

solver_table.jac_evals_saved = ...
    solver_table.STD_jac_evals - solver_table.GNN_jac_evals;

solver_table.linear_solves_saved = ...
    solver_table.STD_linear_solves - solver_table.GNN_linear_solves;

solver_table.time_saved = ...
    solver_table.STD_time - solver_table.GNN_time;

solver_table.iteration_reduction_percent = ...
    100 * solver_table.iterations_saved ...
    ./ solver_table.STD_iterations;

solver_table.jac_reduction_percent = ...
    100 * solver_table.jac_evals_saved ...
    ./ solver_table.STD_jac_evals;

solver_table.linear_solves_reduction_percent = ...
    100 * solver_table.linear_solves_saved ...
    ./ solver_table.STD_linear_solves;

solver_table.speedup = ...
    solver_table.STD_time ./ solver_table.GNN_time;

results.solver_table = solver_table;


excel_file = 'solver_comparison.xlsx';

writetable(solver_table, excel_file);

fprintf('\nSolver comparison table saved to:\n%s\n', excel_file);

%% ============================================================
% 8. PLOT SOLVER COMPARISON
% ============================================================

plot_nonlinear_iterations(solver_table);
end
