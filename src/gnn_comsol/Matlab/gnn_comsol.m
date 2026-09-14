clear;
clc;

import com.comsol.model.*
import com.comsol.model.util.*

model_file = ...
    '\\nl-filer1\users$\giovanni\Desktop\Comsol simulations\channel2d_smoother_geometric_variables.mph';

gnn_file = ...
    'C:\Users\giovanni\.comsol\v64\llmatlab\gnn_predictions.mat';

model = mphload(model_file);
G = load(gnn_file);

k = 1;

u_pred = G.u_pred(k,:).';
v_pred = G.v_pred(k,:).';
p_pred = G.p_pred(k,:).';

dt = G.delta_t(k);

fprintf('Prediction %d\n', k);
fprintf('dt = %.15e s\n', dt);

U_template = mphgetu(model, ...
    'soltag', 'sol1', ...
    'solnum', 1);

U_guess = U_template;


%% ============================================================
% DIAGNOSTIC: COMSOL SOLUTION STRUCTURE
% ============================================================

fprintf('\n========================================\n');
fprintf('SOLUTION INFORMATION\n');
fprintf('========================================\n');

%% sol1
fprintf('\n--- sol1 ---\n');

info1 = mphsolinfo(model, 'soltag', 'sol1');
disp(info1);

U1 = mphgetu(model, ...
    'soltag', 'sol1', ...
    'solnum', 1);

fprintf('length(U1) = %d\n', length(U1));


%% sol3
fprintf('\n--- sol3 ---\n');

try
    info3 = mphsolinfo(model, 'soltag', 'sol3');
    disp(info3);
catch ME
    fprintf('sol3 does not contain a solution yet:\n%s\n', ME.message);
end


%% Extended mesh / DOF information
fprintf('\n--- Extended mesh ---\n');

mesh_info = mphxmeshinfo(model);

fprintf('Number of DOFs = %d\n', mesh_info.ndofs);

fprintf('\nDOF names:\n');
disp(mesh_info.dofs.dofnames);


%% Solver tags
fprintf('\n--- sol1 features ---\n');
disp(cell(model.sol('sol1').feature().tags()));

fprintf('\n--- sol3 features ---\n');
disp(cell(model.sol('sol3').feature().tags()));