%% Note that this script is just explainatory, it is not running in this folder, it needs to run through the Comsol LiveLink with MATLAB

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

clear all
clc
close all


%% 0.1 COMSOL LiveLink initialization

import com.comsol.model.*
import com.comsol.model.util.*


%% 0.2 User configuration

% COMSOL model
model_file = ...
    '\\nl-filer1\users$\giovanni\Desktop\Comsol simulations\channel2d_elliptic_cavity.mph';

% COMSOL solution dataset
dataset_tag = 'dset1';

% COMSOL dependent variables
u_var = 'u';
v_var = 'v';
p_var = 'p';

% Output dataset
output_file = 'channel2d_elliptic_cavity_gnn_dataset.mat';


%% 0.3 Load COMSOL model

model = mphload(model_file);

fprintf('\n========================================\n');
fprintf('COMSOL model loaded successfully.\n');
fprintf('Model file:\n%s\n', model_file);
fprintf('========================================\n\n');


%% ============================================================
%  1. MESH
% ============================================================

[stats, meshdata] = mphmeshstats(model);

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

info = mphsolinfo(model);

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
%
% Saved variables:
%
% X
%   Nt x N x 3
%   Node features [u,v,p] for every timestep
%
% edge_index
%   2*Nedges x 2 in the MATLAB file
%   Each row = [source,target]
%   Already converted to zero-based Python indexing
%
% edge_weight
%   2*Nedges x 1
%   Geometric weight associated with each directed edge
%
% P
%   2 x N
%   Mesh-node coordinates
%
% t
%   Time vector
%
% h
%   Global mesh size
% ============================================================

save(output_file, ...
    'X', ...
    'edge_index', ...
    'edge_weight', ...
    'P', ...
    't', ...
    'h', ...
    '-v7.3');


fprintf('\nDataset saved successfully:\n%s\n', output_file);

fprintf('\nCOMSOL -> MATLAB -> PyTorch preprocessing completed.\n');


fprintf('\nMesh element types:\n');

for k = 1:length(meshdata.types)

    fprintf('%s : %d elements\n', ...
        meshdata.types{k}, ...
        size(meshdata.elem{k},2));

end