%% ============================================================
% GNN -> COMSOL INITIAL GUESS
%
% One-step Time Dependent test
%
% Goal:
%   1. Load one GNN prediction
%   2. Pass u_GNN, v_GNN, p_GNN to COMSOL
%   3. Use them as Initial Values for sol3
%   4. Solve one BDF1 time step
%
% COMSOL model already contains:
%
%   Global Definitions
%       int4 -> uGNN(x,y)
%       int5 -> vGNN(x,y)
%       int6 -> pGNN(x,y)
%
%   Laminar Flow -> Initial Values 1
%       u = uGNN(x,y)
%       v = vGNN(x,y)
%       p = pGNN(x,y)
%
% ============================================================

clear;
clc;
close all;

import com.comsol.model.*
import com.comsol.model.util.*


%% ============================================================
% 1. FILE PATHS
% ============================================================

model_file = ...
    '\\nl-filer1\users$\giovanni\Desktop\Comsol simulations\channel2d_gnn_initial_guess_test.mph';

gnn_file = ...
    'C:\Users\giovanni\.comsol\v64\llmatlab\gnn_predictions.mat';

u_file = ...
    'C:\Users\giovanni\.comsol\v64\llmatlab\u_gnn.txt';

v_file = ...
    'C:\Users\giovanni\.comsol\v64\llmatlab\v_gnn.txt';

p_file = ...
    'C:\Users\giovanni\.comsol\v64\llmatlab\p_gnn.txt';


%% ============================================================
% 2. LOAD COMSOL MODEL
% ============================================================

model = mphload(model_file);

fprintf('\nCOMSOL model loaded.\n');


%% ============================================================
% 3. LOAD GNN PREDICTIONS
% ============================================================

G = load(gnn_file);

num_samples = size(G.u_pred, 1);
num_nodes   = size(G.u_pred, 2);

fprintf('GNN predictions loaded.\n');
fprintf('Number of predictions: %d\n', num_samples);
fprintf('Number of nodes:       %d\n', num_nodes);


%% ============================================================
% 4. SELECT GNN SAMPLE
% ============================================================

% For now test only one transition
k = 1;

% Coordinates of GNN nodes
x = G.node_coordinates(:,1);
y = G.node_coordinates(:,2);

% GNN prediction for the next state
u_pred = G.u_pred(k,:).';
v_pred = G.v_pred(k,:).';
p_pred = G.p_pred(k,:).';

% Time step associated with this transition
dt = G.delta_t(k);


fprintf('\n========================================\n');
fprintf('GNN SAMPLE %d\n', k);
fprintf('========================================\n');

fprintf('Nodes = %d\n', num_nodes);
fprintf('dt    = %.15e s\n', dt);


%% ============================================================
% 5. WRITE GNN PREDICTION TO FILES
%
% Each file contains:
%
%       x    y    predicted_value
%
% ============================================================

writematrix( ...
    [x, y, u_pred], ...
    u_file, ...
    'Delimiter', 'space');

writematrix( ...
    [x, y, v_pred], ...
    v_file, ...
    'Delimiter', 'space');

writematrix( ...
    [x, y, p_pred], ...
    p_file, ...
    'Delimiter', 'space');


fprintf('\nGNN prediction written to interpolation files.\n');


%% ============================================================
% 6. REFRESH COMSOL INTERPOLATION FUNCTIONS
%
% int4 -> uGNN(x,y)
% int5 -> vGNN(x,y)
% int6 -> pGNN(x,y)
%
% ============================================================

model.func('int4').refresh();
model.func('int5').refresh();
model.func('int6').refresh();

fprintf('COMSOL GNN interpolation functions refreshed.\n');


%% ============================================================
% 7. CHECK PHYSICS INITIAL VALUES
%
% These were already configured in the COMSOL model:
%
%   u = uGNN(x,y)
%   v = vGNN(x,y)
%   p = pGNN(x,y)
%
% ============================================================

init1 = ...
    model.component('comp1').physics('spf').feature('init1');

Pinit = mphgetproperties(init1);

fprintf('\n========================================\n');
fprintf('PHYSICS INITIAL VALUES\n');
fprintf('========================================\n');

fprintf('Velocity = %s\n', Pinit.u_init);
fprintf('Pressure = %s\n', Pinit.p_init);


%% ============================================================
% 8. CONFIGURE SOL3 TO USE PHYSICS INITIAL VALUES
% ============================================================

v1 = model.sol('sol3').feature('v1');

% Use Initial Values defined in the physics
v1.set('initmethod', 'init');


%% ============================================================
% 9. CONFIGURE ONE TIME STEP
%
% Use:
%
%   BDF
%   order = 1
%   dt = delta_t(k)
%   t = 0 -> dt
%
% ============================================================

t1 = model.sol('sol3').feature('t1');

dt_str = sprintf('%.17g', dt);

% BDF time integration
t1.set('timemethod', 'bdf');

% Manual time stepping
t1.set('tstepsbdf', 'manual');

% BDF1 = Backward Euler
t1.set('maxorder', '1');

% Time-step size
t1.set('timestepbdf', dt_str);

% Solve exactly one interval
t1.set('tlist', ['0 ' dt_str]);


fprintf('\n========================================\n');
fprintf('SOL3 CONFIGURATION\n');
fprintf('========================================\n');

fprintf('Sample      = %d\n', k);
fprintf('dt          = %.15e s\n', dt);
fprintf('tlist       = %s\n', ...
    char(t1.getString('tlist')));
fprintf('timemethod  = %s\n', ...
    char(t1.getString('timemethod')));
fprintf('tstepsbdf   = %s\n', ...
    char(t1.getString('tstepsbdf')));
fprintf('maxorder    = %s\n', ...
    char(t1.getString('maxorder')));


%% ============================================================
% 10. RUN SOL3
% ============================================================

fprintf('\n========================================\n');
fprintf('RUNNING SOL3\n');
fprintf('========================================\n');

tic;

try

    model.sol('sol3').runAll();

    elapsed_time = toc;

    fprintf('\n========================================\n');
    fprintf('SOL3 CONVERGED\n');
    fprintf('========================================\n');

    fprintf('Sample       = %d\n', k);
    fprintf('dt           = %.15e s\n', dt);
    fprintf('Elapsed time = %.6f s\n', elapsed_time);

catch ME

    elapsed_time = toc;

    fprintf('\n========================================\n');
    fprintf('SOL3 FAILED\n');
    fprintf('========================================\n');

    fprintf('Sample       = %d\n', k);
    fprintf('dt           = %.15e s\n', dt);
    fprintf('Elapsed time = %.6f s\n\n', elapsed_time);

    fprintf('%s\n', ME.message);

end