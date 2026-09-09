%% ============================================================
% PRESSURE STATISTICAL AND BOUNDARY ANALYSIS
% ============================================================




%% ============================================================
% CHECK DATA
% ============================================================

fprintf("\n========================================\n");
fprintf("DATASET INFORMATION\n");
fprintf("========================================\n");

fprintf("Shape of X:\n");
disp(size(X));

fprintf("Shape of P:\n");
disp(size(P));


%% ============================================================
% EXTRACT PRESSURE
% ============================================================

% Assumption:
%
% X -> [num_timesteps, num_nodes, 3]
%
% X(:,:,1) = u
% X(:,:,2) = v
% X(:,:,3) = p

p = X(:,:,3);

fprintf("\nPressure shape:\n");
disp(size(p));

nt = size(p, 1);
n_nodes = size(p, 2);


%% ============================================================
% NODE COORDINATES
% ============================================================

% We need:
%
% pos -> [num_nodes, 2]
%
% Depending on how P was saved, it could be:
%
% [num_nodes, 2]
%
% or:
%
% [2, num_nodes]

if size(P,1) == n_nodes && size(P,2) >= 2

    pos = P(:,1:2);

elseif size(P,2) == n_nodes && size(P,1) >= 2

    pos = P(1:2,:)';

else

    error( ...
        "Cannot determine node coordinates from P. " + ...
        "Pressure has %d nodes, while size(P) = [%d %d].", ...
        n_nodes, ...
        size(P,1), ...
        size(P,2) ...
    );

end

x = pos(:,1);
y = pos(:,2);

fprintf("\nCoordinate matrix shape:\n");
disp(size(pos));


%% ============================================================
% FLATTEN PRESSURE
% ============================================================

p_all = p(:);


%% ============================================================
% BASIC STATISTICS
% ============================================================

p_mean   = mean(p_all);
p_std    = std(p_all);
p_median = median(p_all);

p_min = min(p_all);
p_max = max(p_all);


%% ============================================================
% PERCENTILES
% ============================================================

p01  = prctile(p_all, 0.1);
p1   = prctile(p_all, 1);
p5   = prctile(p_all, 5);

q25  = prctile(p_all, 25);
q75  = prctile(p_all, 75);

p95  = prctile(p_all, 95);
p99  = prctile(p_all, 99);
p999 = prctile(p_all, 99.9);

p_iqr = q75 - q25;


%% ============================================================
% SKEWNESS AND KURTOSIS WITHOUT TOOLBOX
% ============================================================

centered = p_all - p_mean;

p_skewness = ...
    mean(centered.^3) / (p_std^3);

p_kurtosis = ...
    mean(centered.^4) / (p_std^4);


%% ============================================================
% PRINT PRESSURE STATISTICS
% ============================================================

fprintf("\n");
fprintf("========================================\n");
fprintf("PRESSURE STATISTICS\n");
fprintf("========================================\n");

fprintf("Mean:       %.6e\n", p_mean);
fprintf("Std:        %.6e\n", p_std);
fprintf("Median:     %.6e\n", p_median);

fprintf("\n");

fprintf("Min:        %.6e\n", p_min);
fprintf("Max:        %.6e\n", p_max);

fprintf("\n");

fprintf("P0.1:       %.6e\n", p01);
fprintf("P1:         %.6e\n", p1);
fprintf("P5:         %.6e\n", p5);

fprintf("P25:        %.6e\n", q25);
fprintf("P75:        %.6e\n", q75);

fprintf("P95:        %.6e\n", p95);
fprintf("P99:        %.6e\n", p99);
fprintf("P99.9:      %.6e\n", p999);

fprintf("\n");

fprintf("IQR:        %.6e\n", p_iqr);
fprintf("Skewness:   %.6e\n", p_skewness);
fprintf("Kurtosis:   %.6e\n", p_kurtosis);

fprintf("\n");

fprintf("Std / IQR:       %.6f\n", p_std / p_iqr);
fprintf("Max / P99:       %.6f\n", p_max / p99);
fprintf("Max / P99.9:     %.6f\n", p_max / p999);

fprintf("========================================\n");


%% ============================================================
% FULL PRESSURE HISTOGRAM
% ============================================================

figure;

histogram(p_all, 200);

xlabel("Pressure");
ylabel("Count");

title("Pressure Distribution - Full Dataset");

set(gca, "YScale", "log");

grid on;


%% ============================================================
% CENTRAL DISTRIBUTION: P1 - P99
% ============================================================

p_central = p_all( ...
    p_all >= p1 & ...
    p_all <= p99 ...
);

figure;

histogram(p_central, 200);

xlabel("Pressure");
ylabel("Count");

title("Pressure Distribution - P1 to P99");

grid on;


%% ============================================================
% CENTRAL DISTRIBUTION: P0.1 - P99.9
% ============================================================

p_central_999 = p_all( ...
    p_all >= p01 & ...
    p_all <= p999 ...
);

figure;

histogram(p_central_999, 200);

xlabel("Pressure");
ylabel("Count");

title("Pressure Distribution - P0.1 to P99.9");

grid on;


%% ============================================================
% PRESSURE STATISTICS PER TIMESTEP
% ============================================================

p_min_t = min(p, [], 2);
p_max_t = max(p, [], 2);

p_mean_t = mean(p, 2);
p_median_t = median(p, 2);

p_min_t = squeeze(p_min_t);
p_max_t = squeeze(p_max_t);

p_mean_t = squeeze(p_mean_t);
p_median_t = squeeze(p_median_t);


%% ============================================================
% FIND GLOBAL MAXIMUM AND MINIMUM
% ============================================================

[global_max, linear_max_idx] = max(p(:));

[timestep_max, node_max] = ind2sub( ...
    size(p), ...
    linear_max_idx ...
);

[global_min, linear_min_idx] = min(p(:));

[timestep_min, node_min] = ind2sub( ...
    size(p), ...
    linear_min_idx ...
);

fprintf("\n");
fprintf("========================================\n");
fprintf("GLOBAL PRESSURE EXTREMES\n");
fprintf("========================================\n");

fprintf( ...
    "Maximum pressure = %.6e\n", ...
    global_max ...
);

fprintf( ...
    "Maximum timestep = %d\n", ...
    timestep_max ...
);

fprintf( ...
    "Maximum node     = %d\n", ...
    node_max ...
);

fprintf( ...
    "Coordinates      = (%.6e, %.6e)\n", ...
    x(node_max), ...
    y(node_max) ...
);

fprintf("\n");

fprintf( ...
    "Minimum pressure = %.6e\n", ...
    global_min ...
);

fprintf( ...
    "Minimum timestep = %d\n", ...
    timestep_min ...
);

fprintf( ...
    "Minimum node     = %d\n", ...
    node_min ...
);

fprintf( ...
    "Coordinates      = (%.6e, %.6e)\n", ...
    x(node_min), ...
    y(node_min) ...
);

fprintf("========================================\n");


%% ============================================================
% PRESSURE EXTREMES OVER TIME
% ============================================================

figure;

plot( ...
    p_max_t, ...
    "DisplayName", "max(p)" ...
);

hold on;

plot( ...
    p_min_t, ...
    "DisplayName", "min(p)" ...
);

xlabel("Timestep");
ylabel("Pressure");

title("Pressure Extremes Over Time");

legend("Location", "best");

grid on;


%% ============================================================
% MEAN AND MEDIAN OVER TIME
% ============================================================

figure;

plot( ...
    p_mean_t, ...
    "DisplayName", "mean(p)" ...
);

hold on;

plot( ...
    p_median_t, ...
    "DisplayName", "median(p)" ...
);

xlabel("Timestep");
ylabel("Pressure");

title("Mean and Median Pressure Over Time");

legend("Location", "best");

grid on;


%% ============================================================
% LOG SCALE: MAX, MEAN, MEDIAN
% ============================================================

figure;

semilogy( ...
    p_max_t, ...
    "DisplayName", "max(p)" ...
);

hold on;

semilogy( ...
    p_mean_t, ...
    "DisplayName", "mean(p)" ...
);

semilogy( ...
    p_median_t, ...
    "DisplayName", "median(p)" ...
);

xlabel("Timestep");
ylabel("Pressure");

title("Pressure Statistics Over Time - Log Scale");

legend("Location", "best");

grid on;


%% ============================================================
% IDENTIFY CHANNEL BOUNDARIES
% ============================================================

xmin = min(x);
xmax = max(x);

ymin = min(y);
ymax = max(y);

Lx = xmax - xmin;
Ly = ymax - ymin;

% Relative geometric tolerance.
% Increase this if some boundary nodes are not detected.
tol = 1e-8 * max(Lx, Ly);

inlet_nodes = ...
    abs(x - xmin) <= tol;

outlet_nodes = ...
    abs(x - xmax) <= tol;

bottom_nodes = ...
    abs(y - ymin) <= tol;

top_nodes = ...
    abs(y - ymax) <= tol;

wall_nodes = ...
    bottom_nodes | top_nodes;

boundary_nodes = ...
    inlet_nodes | ...
    outlet_nodes | ...
    wall_nodes;

interior_nodes = ~boundary_nodes;


%% ============================================================
% PRINT BOUNDARY INFORMATION
% ============================================================

fprintf("\n");
fprintf("========================================\n");
fprintf("BOUNDARY NODES\n");
fprintf("========================================\n");

fprintf("xmin = %.6e\n", xmin);
fprintf("xmax = %.6e\n", xmax);
fprintf("ymin = %.6e\n", ymin);
fprintf("ymax = %.6e\n", ymax);

fprintf("\n");

fprintf( ...
    "Inlet nodes:       %d\n", ...
    sum(inlet_nodes) ...
);

fprintf( ...
    "Outlet nodes:      %d\n", ...
    sum(outlet_nodes) ...
);

fprintf( ...
    "Top wall nodes:    %d\n", ...
    sum(top_nodes) ...
);

fprintf( ...
    "Bottom wall nodes: %d\n", ...
    sum(bottom_nodes) ...
);

fprintf( ...
    "All boundary:      %d\n", ...
    sum(boundary_nodes) ...
);

fprintf( ...
    "Interior:          %d\n", ...
    sum(interior_nodes) ...
);

fprintf("========================================\n");


%% ============================================================
% VISUALIZE IDENTIFIED BOUNDARIES
% ============================================================

figure;

scatter( ...
    x(interior_nodes), ...
    y(interior_nodes), ...
    5, ...
    "filled", ...
    "DisplayName", "Interior" ...
);

hold on;

scatter( ...
    x(inlet_nodes), ...
    y(inlet_nodes), ...
    20, ...
    "filled", ...
    "DisplayName", "Inlet" ...
);

scatter( ...
    x(outlet_nodes), ...
    y(outlet_nodes), ...
    20, ...
    "filled", ...
    "DisplayName", "Outlet" ...
);

scatter( ...
    x(wall_nodes), ...
    y(wall_nodes), ...
    20, ...
    "filled", ...
    "DisplayName", "Walls" ...
);

xlabel("x");
ylabel("y");

title("Detected Boundary Nodes");

legend("Location", "best");

axis equal;

grid on;


%% ============================================================
% CLASSIFY GLOBAL MAXIMUM
% ============================================================

fprintf("\n");
fprintf("========================================\n");
fprintf("GLOBAL MAXIMUM CLASSIFICATION\n");
fprintf("========================================\n");

fprintf( ...
    "p_max = %.6e\n", ...
    global_max ...
);

fprintf( ...
    "Timestep = %d\n", ...
    timestep_max ...
);

fprintf( ...
    "Node = %d\n", ...
    node_max ...
);

fprintf( ...
    "Coordinates = (%.6e, %.6e)\n", ...
    x(node_max), ...
    y(node_max) ...
);

fprintf("\n");

fprintf( ...
    "Inlet:       %d\n", ...
    inlet_nodes(node_max) ...
);

fprintf( ...
    "Outlet:      %d\n", ...
    outlet_nodes(node_max) ...
);

fprintf( ...
    "Top wall:    %d\n", ...
    top_nodes(node_max) ...
);

fprintf( ...
    "Bottom wall: %d\n", ...
    bottom_nodes(node_max) ...
);

fprintf( ...
    "Interior:    %d\n", ...
    interior_nodes(node_max) ...
);

fprintf("========================================\n");


%% ============================================================
% EXTRACT PRESSURE AT DIFFERENT REGIONS
% ============================================================

p_inlet = p(:, inlet_nodes);
p_outlet = p(:, outlet_nodes);

p_top = p(:, top_nodes);
p_bottom = p(:, bottom_nodes);

p_walls = p(:, wall_nodes);

p_boundary = p(:, boundary_nodes);
p_interior = p(:, interior_nodes);


%% ============================================================
% MEAN PRESSURE AT BOUNDARIES OVER TIME
% ============================================================

p_inlet_mean = mean(p_inlet, 2);
p_outlet_mean = mean(p_outlet, 2);

p_top_mean = mean(p_top, 2);
p_bottom_mean = mean(p_bottom, 2);

figure;

plot( ...
    p_inlet_mean, ...
    "DisplayName", "Inlet" ...
);

hold on;

plot( ...
    p_outlet_mean, ...
    "DisplayName", "Outlet" ...
);

plot( ...
    p_top_mean, ...
    "DisplayName", "Top wall" ...
);

plot( ...
    p_bottom_mean, ...
    "DisplayName", "Bottom wall" ...
);

xlabel("Timestep");
ylabel("Mean Pressure");

title("Mean Pressure at Boundaries");

legend("Location", "best");

grid on;


%% ============================================================
% MAXIMUM PRESSURE AT BOUNDARIES OVER TIME
% ============================================================

p_inlet_max = max(p_inlet, [], 2);
p_outlet_max = max(p_outlet, [], 2);

p_top_max = max(p_top, [], 2);
p_bottom_max = max(p_bottom, [], 2);

figure;

plot( ...
    p_inlet_max, ...
    "DisplayName", "Inlet" ...
);

hold on;

plot( ...
    p_outlet_max, ...
    "DisplayName", "Outlet" ...
);

plot( ...
    p_top_max, ...
    "DisplayName", "Top wall" ...
);

plot( ...
    p_bottom_max, ...
    "DisplayName", "Bottom wall" ...
);

xlabel("Timestep");
ylabel("Maximum Pressure");

title("Maximum Pressure at Boundaries");

legend("Location", "best");

grid on;


%% ============================================================
% BOUNDARY VS INTERIOR PRESSURE
% ============================================================

p_boundary_max = max(p_boundary, [], 2);
p_interior_max = max(p_interior, [], 2);

figure;

plot( ...
    p_boundary_max, ...
    "DisplayName", "Boundary max" ...
);

hold on;

plot( ...
    p_interior_max, ...
    "DisplayName", "Interior max" ...
);

xlabel("Timestep");
ylabel("Maximum Pressure");

title("Boundary vs Interior Maximum Pressure");

legend("Location", "best");

grid on;


%% ============================================================
% DISTRIBUTION: BOUNDARY VS INTERIOR
% ============================================================

figure;

histogram( ...
    p_boundary(:), ...
    200, ...
    "DisplayName", "Boundary" ...
);

hold on;

histogram( ...
    p_interior(:), ...
    200, ...
    "DisplayName", "Interior" ...
);

xlabel("Pressure");
ylabel("Count");

title("Pressure Distribution - Boundary vs Interior");

set(gca, "YScale", "log");

legend("Location", "best");

grid on;


%% ============================================================
% PRESSURE FIELD AT GLOBAL MAXIMUM TIMESTEP
% ============================================================

figure;

scatter( ...
    x, ...
    y, ...
    15, ...
    p(timestep_max,:), ...
    "filled" ...
);

hold on;

scatter( ...
    x(node_max), ...
    y(node_max), ...
    100, ...
    "x", ...
    "LineWidth", 2 ...
);

xlabel("x");
ylabel("y");

title( ...
    sprintf( ...
        "Pressure Field at Timestep %d - Global Maximum", ...
        timestep_max ...
    ) ...
);

colorbar;

axis equal;

grid on;


%% ============================================================
% FIRST 20 TIMESTEPS
% ============================================================

fprintf("\n");
fprintf("========================================\n");
fprintf("FIRST 20 TIMESTEPS\n");
fprintf("========================================\n");

fprintf( ...
    "t\tmin\t\tmean\t\tmedian\t\tmax\n" ...
);

for ti = 1:min(20, nt)

    fprintf( ...
        "%d\t%.4e\t%.4e\t%.4e\t%.4e\n", ...
        ti, ...
        p_min_t(ti), ...
        p_mean_t(ti), ...
        p_median_t(ti), ...
        p_max_t(ti) ...
    );

end

fprintf("========================================\n");