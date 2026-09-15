
function plot_spatial_error(results, k, variable, method)

% ============================================================
% PLOT SPATIAL ERROR
%
% Visualizes the node-wise absolute error on the 2D COMSOL mesh.
%
% Example:
%
%   plot_spatial_error(results, 1000, 'p', 'standard')
%
%   plot_spatial_error(results, 1000, 'p', 'gnn')
%
% ============================================================


%% Find requested transition

idx = find(results.k == k, 1);

if isempty(idx)
    error('Transition k = %d is not present in results.', k);
end


%% Node coordinates

P = results.node_coordinates;

x = P(:,1);
y = P(:,2);


%% Select error field

switch lower(method)

    case 'standard'

        switch lower(variable)

            case 'u'
                err = results.error_standard_td_u{idx};

            case 'v'
                err = results.error_standard_td_v{idx};

            case 'p'
                err = results.error_standard_td_p{idx};

            otherwise
                error('Unknown variable: %s', variable);
        end

        method_name = 'Stationary Standard vs Time Dependent';


    case 'gnn'

        switch lower(variable)

            case 'u'
                err = results.error_gnn_td_u{idx};

            case 'v'
                err = results.error_gnn_td_v{idx};

            case 'p'
                err = results.error_gnn_td_p{idx};

            otherwise
                error('Unknown variable: %s', variable);
        end

        method_name = 'Stationary GNN vs Time Dependent';


    otherwise

        error('Unknown method: %s', method);

end


%% Plot spatial field

figure;

scatter(x, y, 18, err, 'filled');

axis equal;
axis tight;

colorbar;

xlabel('x [m]');
ylabel('y [m]');

title(sprintf( ...
    '%s | %s error | k = %d | t = %.6f s', ...
    method_name, ...
    variable, ...
    k, ...
    results.t_target(idx)));

end
