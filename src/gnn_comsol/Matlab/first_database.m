
%% ============================================================
%  COMSOL -> GNN DATASET
%
%  Purpose:
%  1. Load an existing COMSOL model
%  2. Extract the geometric mesh
%  3. Evaluate u, v, p at all mesh vertices and all timesteps
%  4. Build the GNN node-feature tensor X
%  5. Extract mesh connectivity
%  6. Build graph edges
%  7. Compute geometric edge weights
%  8. Prepare edge_index and edge_weight for PyTorch Geometric
%  9. Save the dataset
%
%  Node features:
%
%       X(n,i,:) = [u_i^n, v_i^n, p_i^n]
%
%  where:
%       n = timestep
%       i = graph/mesh node
%
%  Edge weight:
%
%       w_ij = 1 / (1 + d_ij/h)
%
%  IMPORTANT:
%  Self-loops and GCN normalization are NOT performed here.
%  They will be handled in Python by PyTorch Geometric GCNConv.
% ============================================================


%% 0. Initialization

clc
close all


%% 0.1 COMSOL LiveLink initialization

import com.comsol.model.*
import com.comsol.model.util.*


%% 0.2 User configuration

% COMSOL model
model_file = ...
    '\\nl-filer1\users$\giovanni\Desktop\Comsol simulations\Reynolds dataset\channel2d_manual_stabilization_200R.mph';


% Run COMSOL simulation?
% true  -> run Study 1
% false -> use the solution already stored in the .mph file
run_simulation = false;

% COMSOL time-dependent solution
solution_tag = 'sol4';

% COMSOL solution dataset
dataset_tag = 'dset4';

% COMSOL dependent variables
u_var = 'u';
v_var = 'v';
p_var = 'p';

% COMSOL physics-derived variables
du_dx_var   = 'du_dx';
du_dy_var   = 'du_dy';
dv_dx_var   = 'dv_dx';
dv_dy_var   = 'dv_dy';
div_conv_var = 'div_conv';

% Output dataset
output_file = ...
    'C:\Users\giovanni\.comsol\v64\llmatlab\channel2d_manual_stabilization_200R.mat';


%% 0.3 Load COMSOL model

model = mphload(model_file);

fprintf('\n========================================\n');
fprintf('COMSOL model loaded successfully.\n');
fprintf('Model file:\n%s\n', model_file);
fprintf('========================================\n\n');


%% ============================================================
% 0.4 RUN STUDY 1 - TIME DEPENDENT
% ============================================================

if run_simulation

    fprintf('\n========================================\n');
    fprintf('RUNNING STUDY 1 - TIME DEPENDENT\n');
    fprintf('========================================\n');

    tic;

    try

        model.study('std1').run;

        elapsed_time = toc;

        fprintf('\n========================================\n');
        fprintf('STUDY 1 CONVERGED\n');
        fprintf('========================================\n');

        fprintf('Elapsed time = %.6f s\n', elapsed_time);

    catch ME

        elapsed_time = toc;

        fprintf('\n========================================\n');
        fprintf('STUDY 1 FAILED\n');
        fprintf('========================================\n');

        fprintf('Elapsed time = %.6f s\n\n', elapsed_time);
        fprintf('%s\n', ME.message);

        rethrow(ME);

    end

else

    fprintf('\n========================================\n');
    fprintf('SKIPPING STUDY 1\n');
    fprintf('Using solution already stored in COMSOL model.\n');
    fprintf('========================================\n');

end


%% ============================================================
%  1. MESH
% ============================================================

[stats, meshdata] = mphmeshstats(model, "mesh1");

% Coordinates of geometric mesh vertices:
%
% P(1,:) = x coordinates
% P(2,:) = y coordinates
%
% Therefore:
%
% P(:,i) = [x_i; y_i]

P = meshdata.vertex;

% Number of geometric mesh vertices
N = size(P,2);

fprintf('Number of mesh nodes: %d\n', N);


%% ============================================================
%  2. TIME INFORMATION
% ============================================================

info = mphsolinfo(model, 'soltag', solution_tag);

% Time vector
t = info.solvals;

% Number of timesteps
Nt = length(t);

fprintf('Number of timesteps: %d\n', Nt);
fprintf('Initial time: %.6g\n', t(1));
fprintf('Final time:   %.6g\n\n', t(end));


%% ============================================================
%  3. EVALUATE FEM SOLUTION AT MESH NODES
% ============================================================

% Evaluate u at every mesh vertex for every timestep
u_nodes = mphinterp(model, u_var, ...
    'coord', P, ...
    'dataset', dataset_tag, ...
    'solnum', 'all');

% Evaluate v
v_nodes = mphinterp(model, v_var, ...
    'coord', P, ...
    'dataset', dataset_tag, ...
    'solnum', 'all');

% Evaluate p
p_nodes = mphinterp(model, p_var, ...
    'coord', P, ...
    'dataset', dataset_tag, ...
    'solnum', 'all');


%% ============================================================
%  3.2 EVALUATE GEOMETRIC BOUNDARY FEATURES
% ============================================================
%
% For each mesh node, evaluate the reciprocal distance to the
% wall/inlet/outlet boundaries (G2/G3/G4) and the direction
% toward each of them. Combined below into geometry_features.
% ============================================================

geometry_dataset_tag = 'dset2';

% WALL
G_wall = mphinterp(model, 'G2', ...
    'coord', P, ...
    'dataset', geometry_dataset_tag);

wall_dir_x = mphinterp(model, 'wd.Ddirx', ...
    'coord', P, ...
    'dataset', geometry_dataset_tag);

wall_dir_y = mphinterp(model, 'wd.Ddiry', ...
    'coord', P, ...
    'dataset', geometry_dataset_tag);

% INLET
G_inlet = mphinterp(model, 'G3', ...
    'coord', P, ...
    'dataset', geometry_dataset_tag);

inlet_dir_x = mphinterp(model, 'wd2.Ddirx', ...
    'coord', P, ...
    'dataset', geometry_dataset_tag);

inlet_dir_y = mphinterp(model, 'wd2.Ddiry', ...
    'coord', P, ...
    'dataset', geometry_dataset_tag);

% OUTLET
G_outlet = mphinterp(model, 'G4', ...
    'coord', P, ...
    'dataset', geometry_dataset_tag);

outlet_dir_x = mphinterp(model, 'wd3.Ddirx', ...
    'coord', P, ...
    'dataset', geometry_dataset_tag);

outlet_dir_y = mphinterp(model, 'wd3.Ddiry', ...
    'coord', P, ...
    'dataset', geometry_dataset_tag);


%% ============================================================
%  BUILD GEOMETRIC FEATURES
% ============================================================
%
% Static geometric features, one row per graph node.
%
% Columns:
%   1 -> wall geometry x
%   2 -> wall geometry y
%   3 -> inlet geometry x
%   4 -> inlet geometry y
%   5 -> outlet geometry x
%   6 -> outlet geometry y
%
% The reciprocal boundary distance G is combined with the
% direction toward the corresponding boundary.
% ============================================================

wall_geom_x = G_wall .* wall_dir_x;
wall_geom_y = G_wall .* wall_dir_y;

inlet_geom_x = G_inlet .* inlet_dir_x;
inlet_geom_y = G_inlet .* inlet_dir_y;

outlet_geom_x = G_outlet .* outlet_dir_x;
outlet_geom_y = G_outlet .* outlet_dir_y;

% Force column vectors
wall_geom_x = wall_geom_x(:);
wall_geom_y = wall_geom_y(:);

inlet_geom_x = inlet_geom_x(:);
inlet_geom_y = inlet_geom_y(:);

outlet_geom_x = outlet_geom_x(:);
outlet_geom_y = outlet_geom_y(:);

% Build geometric-feature matrix
geometry_features = [
    wall_geom_x, ...
    wall_geom_y, ...
    inlet_geom_x, ...
    inlet_geom_y, ...
    outlet_geom_x, ...
    outlet_geom_y
];

geometry_feature_names = {
    'wall_geom_x', ...
    'wall_geom_y', ...
    'inlet_geom_x', ...
    'inlet_geom_y', ...
    'outlet_geom_x', ...
    'outlet_geom_y'
};

fprintf('\n========================================\n');
fprintf('GEOMETRIC FEATURES\n');
fprintf('========================================\n');

fprintf('geometry_features: %d x %d\n', ...
    size(geometry_features,1), ...
    size(geometry_features,2));


%% ============================================================
%  3.1 EVALUATE PHYSICS FEATURES AT MESH NODES
% ============================================================

% du/dx
du_dx_nodes = mphinterp(model, du_dx_var, ...
    'coord', P, ...
    'dataset', dataset_tag, ...
    'solnum', 'all');

% du/dy
du_dy_nodes = mphinterp(model, du_dy_var, ...
    'coord', P, ...
    'dataset', dataset_tag, ...
    'solnum', 'all');

% dv/dx
dv_dx_nodes = mphinterp(model, dv_dx_var, ...
    'coord', P, ...
    'dataset', dataset_tag, ...
    'solnum', 'all');

% dv/dy
dv_dy_nodes = mphinterp(model, dv_dy_var, ...
    'coord', P, ...
    'dataset', dataset_tag, ...
    'solnum', 'all');

% Divergence of the convective acceleration
%
% div_conv =
% div[(u . grad)u]
%
div_conv_nodes = mphinterp(model, div_conv_var, ...
    'coord', P, ...
    'dataset', dataset_tag, ...
    'solnum', 'all');


%% ============================================================
%  4. CHECK SOLUTION DIMENSIONS
% ============================================================

fprintf('Solution dimensions:\n');

fprintf('P       : %d x %d\n', ...
    size(P,1), size(P,2));

fprintf('u_nodes : %d x %d\n', ...
    size(u_nodes,1), size(u_nodes,2));

fprintf('v_nodes : %d x %d\n', ...
    size(v_nodes,1), size(v_nodes,2));

fprintf('p_nodes : %d x %d\n\n', ...
    size(p_nodes,1), size(p_nodes,2));

% Check consistency
if size(u_nodes,1) ~= Nt || size(u_nodes,2) ~= N
    error('Unexpected dimensions for u_nodes.');
end

if size(v_nodes,1) ~= Nt || size(v_nodes,2) ~= N
    error('Unexpected dimensions for v_nodes.');
end

if size(p_nodes,1) ~= Nt || size(p_nodes,2) ~= N
    error('Unexpected dimensions for p_nodes.');
end

fprintf('du_dx_nodes   : %d x %d\n', ...
    size(du_dx_nodes,1), size(du_dx_nodes,2));

fprintf('du_dy_nodes   : %d x %d\n', ...
    size(du_dy_nodes,1), size(du_dy_nodes,2));

fprintf('dv_dx_nodes   : %d x %d\n', ...
    size(dv_dx_nodes,1), size(dv_dx_nodes,2));

fprintf('dv_dy_nodes   : %d x %d\n', ...
    size(dv_dy_nodes,1), size(dv_dy_nodes,2));

fprintf('div_conv_nodes: %d x %d\n\n', ...
    size(div_conv_nodes,1), size(div_conv_nodes,2));

physics_arrays = {
    du_dx_nodes, ...
    du_dy_nodes, ...
    dv_dx_nodes, ...
    dv_dy_nodes, ...
    div_conv_nodes
};

physics_names = {
    'du_dx_nodes', ...
    'du_dy_nodes', ...
    'dv_dx_nodes', ...
    'dv_dy_nodes', ...
    'div_conv_nodes'
};

for k = 1:length(physics_arrays)

    A = physics_arrays{k};

    if size(A,1) ~= Nt || size(A,2) ~= N
        error( ...
            'Unexpected dimensions for %s.', ...
            physics_names{k} ...
        );
    end

end


%% ============================================================
%  5. BUILD NODE-FEATURE DATASET X
% ============================================================
%
% X dimensions:
%
%       Nt x N x 3
%
% First dimension  -> timestep
% Second dimension -> graph node
% Third dimension  -> physical feature
%
% X(n,i,1) = u_i^n
% X(n,i,2) = v_i^n
% X(n,i,3) = p_i^n
% ============================================================

X = zeros(Nt, N, 3);

X(:,:,1) = u_nodes;
X(:,:,2) = v_nodes;
X(:,:,3) = p_nodes;

fprintf('Node-feature tensor X created.\n');
fprintf('X dimensions: %d x %d x %d\n\n', ...
    size(X,1), size(X,2), size(X,3));


%% ============================================================
%  5.1 BUILD PHYSICS-FEATURE DATASET
% ============================================================
%
% physics_features dimensions:
%
%       Nt x N x 5
%
% physics_features(n,i,1) = du/dx
% physics_features(n,i,2) = du/dy
% physics_features(n,i,3) = dv/dx
% physics_features(n,i,4) = dv/dy
% physics_features(n,i,5) = div[(u . grad)u]
%
% ============================================================

physics_features = zeros(Nt, N, 5);

physics_features(:,:,1) = du_dx_nodes;
physics_features(:,:,2) = du_dy_nodes;
physics_features(:,:,3) = dv_dx_nodes;
physics_features(:,:,4) = dv_dy_nodes;
physics_features(:,:,5) = div_conv_nodes;

% Names corresponding to the third dimension of physics_features
physics_feature_names = {
    'du_dx', ...
    'du_dy', ...
    'dv_dx', ...
    'dv_dy', ...
    'div_conv'
};

fprintf('Physics-feature tensor created.\n');

fprintf( ...
    'physics_features dimensions: %d x %d x %d\n\n', ...
    size(physics_features,1), ...
    size(physics_features,2), ...
    size(physics_features,3) ...
);


%% ============================================================
%  6. EXTRACT TRIANGULAR ELEMENT CONNECTIVITY
% ============================================================
%
% meshdata.types contains the types of mesh entities stored
% by COMSOL.
%
% We search explicitly for triangular elements instead of
% assuming that they are always stored at a fixed position.
% ============================================================

tri_idx = find(strcmp(meshdata.types, 'tri'));

if isempty(tri_idx)
    error('No triangular elements found in mesh.');
end

% Connectivity matrix of triangular elements
T = meshdata.elem{tri_idx(1)};

% COMSOL connectivity may use zero-based indexing:
%
% COMSOL: 0, 1, 2, ...
% MATLAB: 1, 2, 3, ...
%
% Convert only if necessary.

if min(T(:)) == 0
    T = T + 1;
end

% Each column of T represents one triangular element.
%
% T(:,e) = [node_1; node_2; node_3]
%
% Number of triangular elements:

Ne = size(T,2);

fprintf('Number of triangular elements: %d\n', Ne);


%% ============================================================
%  7. COMPUTE GLOBAL MESH SIZE h
% ============================================================
%
% For each triangular element K:
%
%       h_K = maximum edge length of K
%
% Then:
%
%       h = max_K(h_K)
%
% Thus h represents the maximum element diameter according
% to the definition used in this dataset.
% ============================================================

h_elem = zeros(Ne,1);

for e = 1:Ne

    % Nodes belonging to triangle e
    nodes = T(:,e);

    % Coordinates of its three vertices
    p1 = P(:,nodes(1));
    p2 = P(:,nodes(2));
    p3 = P(:,nodes(3));

    % Euclidean lengths of the three triangle edges
    d12 = norm(p1-p2);
    d23 = norm(p2-p3);
    d31 = norm(p3-p1);

    % Element size
    h_elem(e) = max([d12, d23, d31]);

end

% Global mesh size
h = max(h_elem);

fprintf('Global mesh size h: %.6e\n', h);


%% ============================================================
%  8. EXTRACT ALL UNIQUE MESH EDGES
% ============================================================
%
% A triangular element with nodes:
%
%       [n1, n2, n3]
%
% contains the three edges:
%
%       n1 -- n2
%       n2 -- n3
%       n3 -- n1
%
% Adjacent triangles share edges, therefore duplicates
% must be removed.
% ============================================================

% Maximum initial number of edges:
% 3 edges per triangle

edges = zeros(3*Ne,2);

k = 1;

for e = 1:Ne

    nodes = T(:,e);

    edges(k,:)   = [nodes(1), nodes(2)];
    edges(k+1,:) = [nodes(2), nodes(3)];
    edges(k+2,:) = [nodes(3), nodes(1)];

    k = k + 3;

end

% The graph is undirected.
%
% Therefore:
%
%       [i,j]
%
% and
%
%       [j,i]
%
% represent the same geometric mesh edge.
%
% Sorting each row gives a unique representation.

edges = sort(edges,2);

% Remove duplicate mesh edges shared by adjacent triangles
edges = unique(edges,'rows');

% Number of unique undirected graph edges
Nedges = size(edges,1);

fprintf('Number of unique graph edges: %d\n', Nedges);


%% ============================================================
%  9. COMPUTE EUCLIDEAN DISTANCE OF EACH GRAPH EDGE
% ============================================================

% First node of each edge
i = edges(:,1);

% Second node of each edge
j = edges(:,2);

% Coordinate differences
dx = P(1,j) - P(1,i);
dy = P(2,j) - P(2,i);

% Force column vectors
dx = dx(:);
dy = dy(:);

% Euclidean edge distance
dij = sqrt(dx.^2 + dy.^2);

%% ============================================================
%  9.1 PRECOMPUTE WLSQ GEOMETRIC OPERATORS
% ============================================================
%
% For each graph node i:
%
%   1. Find its first-order graph neighbors
%   2. Build the weighted least-squares matrix A_i
%   3. Compute the economy QR factorization
%
% Weight:
%
%       w_ij = 1 / r_ij
%
% where:
%
%       r_ij = sqrt((x_j-x_i)^2 + (y_j-y_i)^2)
%
% The matrix A_i depends ONLY on mesh geometry, therefore it
% can be precomputed once and reused for every timestep.
%
%       A_i * grad(phi_i) ~= b_i
%
% with:
%
%       A_i(row,:) = w_ij * [dx_ij, dy_ij]
%
% and:
%
%       A_i = Q_i * R_i
%
% IMPORTANT:
% These WLSQ weights are different from the edge weights used
% later by the GNN.
% ============================================================


fprintf('\n========================================\n');
fprintf('WLSQ GEOMETRIC PRECOMPUTATION\n');
fprintf('========================================\n');


%% ------------------------------------------------------------
%  Build neighbor list
% ------------------------------------------------------------

neighbors = cell(N,1);

for e = 1:Nedges

    node_i = edges(e,1);
    node_j = edges(e,2);

    % Undirected graph:
    % i is neighbor of j and j is neighbor of i
    neighbors{node_i}(end+1) = node_j;
    neighbors{node_j}(end+1) = node_i;

end


%% ------------------------------------------------------------
%  Allocate WLSQ structures
% ------------------------------------------------------------

A_wlsq = cell(N,1);
Q_wlsq = cell(N,1);
R_wlsq = cell(N,1);
M_wlsq = cell(N,1);
w_wlsq = cell(N,1);
G_wlsq = cell(N,1);

wlsq_rank = zeros(N,1);
wlsq_cond = zeros(N,1);
num_neighbors = zeros(N,1);


%% ------------------------------------------------------------
%  Construct A_i and compute QR for every graph node
% ------------------------------------------------------------

for node_i = 1:N

    neigh = neighbors{node_i};

    % Force column vector
    neigh = neigh(:);

    num_neighbors(node_i) = length(neigh);


    % ---------------------------------------------------------
    % Coordinates of central node
    % ---------------------------------------------------------

    xi = P(1,node_i);
    yi = P(2,node_i);


    % ---------------------------------------------------------
    % Coordinates of neighboring nodes
    % ---------------------------------------------------------

    xj = P(1,neigh).';
    yj = P(2,neigh).';


    % ---------------------------------------------------------
    % Relative coordinates
    %
    % dx = x_j - x_i
    % dy = y_j - y_i
    % ---------------------------------------------------------

    dx_i = xj - xi;
    dy_i = yj - yi;


    % ---------------------------------------------------------
    % Euclidean distance
    % ---------------------------------------------------------

    r_i = sqrt(dx_i.^2 + dy_i.^2);


    % Safety check
    if any(r_i <= eps)

        error( ...
            'Zero or invalid neighbor distance found at node %d.', ...
            node_i ...
        );

    end


    % ---------------------------------------------------------
    % WLSQ weight
    %
    %       w_ij = 1 / r_ij
    % ---------------------------------------------------------

    w_i = 1 ./ r_i;


    % ---------------------------------------------------------
    % Construct WLSQ geometry matrix
    %
    %       A_i =
    %
    %       [ w_i1 dx_i1    w_i1 dy_i1 ]
    %       [ w_i2 dx_i2    w_i2 dy_i2 ]
    %       [     ...            ...    ]
    %
    % Dimensions:
    %
    %       number_of_neighbors x 2
    % ---------------------------------------------------------

    A_i = [
        w_i .* dx_i, ...
        w_i .* dy_i
    ];


    % ---------------------------------------------------------
    % Check that the gradient can be reconstructed
    % ---------------------------------------------------------

    rank_i = rank(A_i);

    if rank_i < 2

        error( ...
            'WLSQ matrix is rank deficient at node %d. Rank = %d.', ...
            node_i, ...
            rank_i ...
        );

    end


    % ---------------------------------------------------------
    % Economy QR decomposition
    %
    %       A_i = Q_i R_i
    %
    % Q_i : k_i x 2
    % R_i : 2 x 2
    % ---------------------------------------------------------

    [Q_i, R_i] = qr(A_i, 0);
    M_i = R_i \ Q_i.';
    G_i = M_i .* w_i.';



    % ---------------------------------------------------------
    % Condition number
    %
    % Since R_i is only 2 x 2, this is extremely cheap.
    % ---------------------------------------------------------

    cond_i = cond(R_i);


    % ---------------------------------------------------------
    % Store results
    % ---------------------------------------------------------

    A_wlsq{node_i} = A_i;
    Q_wlsq{node_i} = Q_i;
    R_wlsq{node_i} = R_i;
    M_wlsq{node_i} = M_i;
    w_wlsq{node_i} = w_i;
    G_wlsq{node_i} = G_i;

    wlsq_rank(node_i) = rank_i;
    wlsq_cond(node_i) = cond_i;

end


%% ------------------------------------------------------------
%  WLSQ diagnostics
% ------------------------------------------------------------

fprintf('WLSQ preprocessing completed.\n\n');

fprintf('Neighbors per node:\n');
fprintf('  min    = %d\n', min(num_neighbors));
fprintf('  mean   = %.2f\n', mean(num_neighbors));
fprintf('  max    = %d\n\n', max(num_neighbors));

fprintf('WLSQ matrix rank:\n');
fprintf('  minimum rank = %d\n\n', min(wlsq_rank));

fprintf('Condition number of R:\n');
fprintf('  min    = %.6e\n', min(wlsq_cond));
fprintf('  mean   = %.6e\n', mean(wlsq_cond));
fprintf('  median = %.6e\n', median(wlsq_cond));
fprintf('  max    = %.6e\n', max(wlsq_cond));


%% ------------------------------------------------------------
%  Show an example node
% ------------------------------------------------------------

example_node = find(num_neighbors == max(num_neighbors), 1);

fprintf('\n========================================\n');
fprintf('WLSQ EXAMPLE - NODE %d\n', example_node);
fprintf('========================================\n');

fprintf('Number of neighbors: %d\n', ...
    num_neighbors(example_node));

fprintf('Neighbors:\n');
disp(neighbors{example_node});

fprintf('A_i:\n');
disp(A_wlsq{example_node});

fprintf('Q_i:\n');
disp(Q_wlsq{example_node});

fprintf('R_i:\n');
disp(R_wlsq{example_node});

fprintf('rank(A_i) = %d\n', ...
    wlsq_rank(example_node));

fprintf('cond(R_i) = %.6e\n', ...
    wlsq_cond(example_node));

%% ============================================================
%  9.2 VALIDATE WLSQ GRADIENT AGAINST COMSOL
% ============================================================

fprintf('\n========================================\n');
fprintf('WLSQ GRADIENT VALIDATION\n');
fprintf('========================================\n');

% Choose one timestep for validation
test_step = 100;

fprintf('Test timestep index: %d\n', test_step);
fprintf('Physical time: %.6e s\n\n', t(test_step));

% COMSOL velocity field at this timestep
u_test = u_nodes(test_step,:).';

% Allocate reconstructed gradients
du_dx_wlsq = zeros(N,1);
du_dy_wlsq = zeros(N,1);


for node_i = 1:N

    % Neighbor indices
    neigh = neighbors{node_i};
    neigh = neigh(:);

    % WLSQ weights previously computed
    w_i = w_wlsq{node_i};

    % Central value
    u_i = u_test(node_i);

    % Neighbor values
    u_j = u_test(neigh);

    % ---------------------------------------------------------
    % Construct RHS:
    %
    % b_i(j) = w_ij * (u_j - u_i)
    % ---------------------------------------------------------

    b_i = w_i .* (u_j - u_i);


    % ---------------------------------------------------------
    % Solve:
    %
    % A_i grad(u_i) ~= b_i
    %
    % A_i = Q_i R_i
    %
    % therefore:
    %
    % R_i grad(u_i) = Q_i' b_i
    % ---------------------------------------------------------

    Q_i = Q_wlsq{node_i};
    R_i = R_wlsq{node_i};
    
    rhs = Q_i.' * b_i;

    grad_u_i = R_i \ rhs;


    % Store gradient components
    du_dx_wlsq(node_i) = grad_u_i(1);
    du_dy_wlsq(node_i) = grad_u_i(2);

end


%% ------------------------------------------------------------
% COMSOL reference
% ------------------------------------------------------------

du_dx_comsol = du_dx_nodes(test_step,:).';
du_dy_comsol = du_dy_nodes(test_step,:).';


%% ------------------------------------------------------------
% Errors
% ------------------------------------------------------------

error_dx = du_dx_wlsq - du_dx_comsol;
error_dy = du_dy_wlsq - du_dy_comsol;

mae_dx = mean(abs(error_dx));
mae_dy = mean(abs(error_dy));

rmse_dx = sqrt(mean(error_dx.^2));
rmse_dy = sqrt(mean(error_dy.^2));


fprintf('du/dx:\n');
fprintf('  MAE  = %.6e\n', mae_dx);
fprintf('  RMSE = %.6e\n\n', rmse_dx);

fprintf('du/dy:\n');
fprintf('  MAE  = %.6e\n', mae_dy);
fprintf('  RMSE = %.6e\n\n', rmse_dy);

rel_l2_dx = norm(error_dx) / norm(du_dx_comsol);
rel_l2_dy = norm(error_dy) / norm(du_dy_comsol);

fprintf('Relative L2 error:\n');
fprintf('  du/dx = %.6e  (%.2f%%)\n', ...
    rel_l2_dx, 100*rel_l2_dx);

fprintf('  du/dy = %.6e  (%.2f%%)\n', ...
    rel_l2_dy, 100*rel_l2_dy);
%% ============================================================
%  10. COMPUTE GEOMETRIC EDGE WEIGHTS
% ============================================================
%
%                  1
%       w_ij = -----------
%               1 + d_ij/h
%

weights = 1 ./ (1 + dij/h);

% Force column vector
weights = weights(:);


%% ============================================================
%  11. PREPARE GRAPH FOR PYTORCH GEOMETRIC
% ============================================================

% Each undirected edge i--j becomes:
%
% i -> j
% j -> i

source = [i; j];
target = [j; i];

% Convert MATLAB indexing (1,...,N)
% to Python indexing (0,...,N-1)
edge_index = [source - 1, target - 1];

% Same weight for both directions
edge_weight = [weights; weights];

cell_index = T.' - 1;

fprintf('cell_index  : %d x %d\n', ...
    size(cell_index,1), ...
    size(cell_index,2));

%% ============================================================
%  PREPARE WLSQ NEIGHBORS FOR PYTHON
% ============================================================

neighbors_python = cell(N,1);

for node_i = 1:N
    neighbors_python{node_i} = neighbors{node_i} - 1;
end

fprintf('neighbors_python : %d cells\n', ...
    length(neighbors_python));
%% ============================================================
%  12. FINAL DATASET CHECKS
% ============================================================

fprintf('\n========================================\n');
fprintf('PYTORCH-READY DATASET\n');
fprintf('========================================\n');

fprintf('X           : %d x %d x %d\n', ...
    size(X,1), size(X,2), size(X,3));

fprintf('edge_index  : %d x %d\n', ...
    size(edge_index,1), size(edge_index,2));

fprintf('edge_weight : %d x %d\n', ...
    size(edge_weight,1), size(edge_weight,2));

fprintf('P           : %d x %d\n', ...
    size(P,1), size(P,2));

fprintf('t           : %d timesteps\n', ...
    length(t));

fprintf('h           : %.6e\n', h);

fprintf('Mesh nodes  : %d\n', N);
fprintf('Triangles   : %d\n', Ne);
fprintf('Unique edges: %d\n', Nedges);

% Consistency checks

if size(edge_index,1) ~= 2*Nedges
    error('Unexpected number of directed edges.');
end

if length(edge_weight) ~= size(edge_index,1)
    error('edge_index and edge_weight are inconsistent.');
end

if any(edge_index(:) < 0) || any(edge_index(:) > N-1)
    error('Invalid Python node indices in edge_index.');
end


%% ============================================================
%  13. SAVE DATASET FOR PYTHON / PYTORCH GEOMETRIC
% ============================================================

save(output_file, ...
    'X', ...
    'physics_features', ...
    'physics_feature_names', ...
    'geometry_features', ...
    'geometry_feature_names', ...
    'edge_index', ...
    'edge_weight', ...
    'T', ...
    'cell_index', ...
    'neighbors_python', ...
    'G_wlsq', ...
    'P', ...
    't', ...
    'h', ...
    '-v7.3');

fprintf('\nDataset saved successfully:\n%s\n', output_file);

fprintf('\nCOMSOL -> MATLAB -> PyTorch preprocessing completed.\n');


%% ============================================================
%  EXTRA INFORMATION
% ============================================================

fprintf('\nMesh element types:\n');

for k = 1:length(meshdata.types)

    fprintf('%s : %d elements\n', ...
        meshdata.types{k}, ...
        size(meshdata.elem{k},2));

end


fprintf('\nPhysics feature ranges:\n');

for k = 1:size(physics_features,3)

    values = physics_features(:,:,k);

    fprintf( ...
        '%-10s | min = % .6e | max = % .6e | NaN = %d | Inf = %d\n', ...
        physics_feature_names{k}, ...
        min(values(:)), ...
        max(values(:)), ...
        sum(isnan(values(:))), ...
        sum(isinf(values(:))) ...
    );

end


fprintf('\nGeometric feature ranges:\n');

for k = 1:size(geometry_features,2)

    values = geometry_features(:,k);

    fprintf( ...
        '%-15s | min = % .6e | max = % .6e | NaN = %d | Inf = %d\n', ...
        geometry_feature_names{k}, ...
        min(values(:)), ...
        max(values(:)), ...
        sum(isnan(values(:))), ...
        sum(isinf(values(:))) ...
    );

end


%% ============================================================
% 14. SAVE COMSOL MODEL WITH TIME-DEPENDENT SOLUTION
% ============================================================

fprintf('\n========================================\n');
fprintf('SAVING COMSOL MODEL\n');
fprintf('========================================\n');

mphsave(model, model_file);

fprintf('COMSOL model saved successfully.\n');
fprintf('Time-dependent solution %s is now stored in:\n%s\n', ...
    solution_tag, model_file);

