

%% ============================================================
% EXTRACT PRESSURE
% =============================================================

% Assumption:
% X has shape:
%
%   [num_timesteps, num_nodes, 3]
%
% X(:,:,1) = u
% X(:,:,2) = v
% X(:,:,3) = p

p = X(:,:,3);

fprintf("\nPressure shape:\n");
disp(size(p));


%% ============================================================
% FLATTEN ALL PRESSURE VALUES
% =============================================================

p_all = p(:);


%% ============================================================
% BASIC STATISTICS
% =============================================================

p_mean   = mean(p_all);
p_std    = std(p_all);
p_median = median(p_all);

p_min = min(p_all);
p_max = max(p_all);

q25 = prctile(p_all, 25);
q75 = prctile(p_all, 75);

p_iqr = q75 - q25;


%% ============================================================
% PERCENTILES
% =============================================================

p01  = prctile(p_all, 0.1);
p1   = prctile(p_all, 1);
p5   = prctile(p_all, 5);

p95  = prctile(p_all, 95);
p99  = prctile(p_all, 99);
p999 = prctile(p_all, 99.9);


%% ============================================================
% ADDITIONAL DISTRIBUTION STATISTICS
% =============================================================

%% ============================================================
% SKEWNESS AND KURTOSIS - WITHOUT TOOLBOX
% =============================================================

centered = p_all - p_mean;

% Third standardized central moment
p_skewness = mean(centered.^3) / (p_std^3);

% Fourth standardized central moment
% Normal distribution -> kurtosis = 3
p_kurtosis = mean(centered.^4) / (p_std^4);


%% ============================================================
% PRINT RESULTS
% =============================================================

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

fprintf("\n");

fprintf("Skewness:   %.6e\n", p_skewness);
fprintf("Kurtosis:   %.6e\n", p_kurtosis);

fprintf("========================================\n");


%% ============================================================
% FULL PRESSURE DISTRIBUTION
% =============================================================

nbins = 200;

figure;

histogram(p_all, nbins);

xlabel("Pressure");
ylabel("Count");

title("Pressure Distribution - Full Dataset");

set(gca, "YScale", "log");

grid on;


%% ============================================================
% CENTRAL PRESSURE DISTRIBUTION
% P1 - P99
% =============================================================

p_lower = p1;
p_upper = p99;

p_central = p_all( ...
    p_all >= p_lower & ...
    p_all <= p_upper ...
);

figure;

histogram(p_central, 200);

xlabel("Pressure");
ylabel("Count");

title("Pressure Distribution - P1 to P99");

grid on;


%% ============================================================
% CENTRAL PRESSURE DISTRIBUTION
% P0.1 - P99.9
%
% Useful to understand whether the extreme maximum is produced
% by a very small number of points.
% =============================================================

p_lower_999 = p01;
p_upper_999 = p999;

p_central_999 = p_all( ...
    p_all >= p_lower_999 & ...
    p_all <= p_upper_999 ...
);

figure;

histogram(p_central_999, 200);

xlabel("Pressure");
ylabel("Count");

title("Pressure Distribution - P0.1 to P99.9");

grid on;


%% ============================================================
% BOXPLOT
% =============================================================

figure;

boxchart(p_all);

ylabel("Pressure");

title("Pressure Boxplot");

grid on;


%% ============================================================
% PRESSURE STATISTICS PER TIMESTEP
% =============================================================

p_min_t    = min(p, [], 2);
p_max_t    = max(p, [], 2);
p_mean_t   = mean(p, 2);
p_median_t = median(p, 2);

p_min_t    = squeeze(p_min_t);
p_max_t    = squeeze(p_max_t);
p_mean_t   = squeeze(p_mean_t);
p_median_t = squeeze(p_median_t);


%% ============================================================
% FIND GLOBAL EXTREME TIMESTEPS
% =============================================================

[global_max, timestep_max] = max(p_max_t);
[global_min, timestep_min] = min(p_min_t);

fprintf("\n");
fprintf("========================================\n");
fprintf("PRESSURE EXTREMES IN TIME\n");
fprintf("========================================\n");

fprintf( ...
    "Global maximum: %.6e at timestep %d\n", ...
    global_max, ...
    timestep_max ...
);

fprintf( ...
    "Global minimum: %.6e at timestep %d\n", ...
    global_min, ...
    timestep_min ...
);

fprintf("========================================\n");


%% ============================================================
% PRESSURE EXTREMES OVER TIME
% =============================================================

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
% MEAN AND MEDIAN PRESSURE OVER TIME
% =============================================================

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
% DIFFERENCE BETWEEN MEAN AND MEDIAN
%
% Large differences indicate asymmetric pressure distributions
% at a particular timestep.
% =============================================================

figure;

plot(p_mean_t - p_median_t);

xlabel("Timestep");
ylabel("Mean(p) - Median(p)");

title("Pressure Asymmetry Over Time");

grid on;


%% ============================================================
% PRESSURE DISTRIBUTION PER TIMESTEP
%
% Use P0.1 - P99.9 instead of min - max.
%
% Otherwise a single extreme value such as p = 205 can compress
% almost the entire useful distribution into a few bins.
% =============================================================

nt = size(p, 1);

heatmap_edges = linspace( ...
    p01, ...
    p999, ...
    201 ...
);

counts = zeros( ...
    nt, ...
    numel(heatmap_edges) - 1 ...
);

for ti = 1:nt

    counts(ti,:) = histcounts( ...
        p(ti,:), ...
        heatmap_edges ...
    );

end


%% Normalize every timestep independently

row_max = max(counts, [], 2);

% Avoid division by zero
row_max(row_max == 0) = 1;

counts_norm = counts ./ row_max;


%% Bin centers

bin_centers = ...
    (heatmap_edges(1:end-1) + heatmap_edges(2:end)) / 2;


%% Plot

figure;

imagesc( ...
    bin_centers, ...
    1:nt, ...
    counts_norm ...
);

axis xy;

colormap(parula);

colorbar;

xlabel("Pressure");
ylabel("Timestep");

title("Pressure Distribution Per Timestep");

hold on;


%% Selected percentiles

xline(p1,  "--w", "P1");
xline(p5,  "--w", "P5");

xline(q25, "--w", "P25");
xline(q75, "--w", "P75");

xline(p95, "--w", "P95");
xline(p99, "--w", "P99");

hold off;


%% ============================================================
% FRACTION OF EXTREME VALUES
% =============================================================

fraction_above_p99 = mean(p_all > p99);
fraction_above_p999 = mean(p_all > p999);

fprintf("\n");
fprintf("========================================\n");
fprintf("TAIL STATISTICS\n");
fprintf("========================================\n");

fprintf( ...
    "Fraction above P99:   %.6f %%\n", ...
    100 * fraction_above_p99 ...
);

fprintf( ...
    "Fraction above P99.9: %.6f %%\n", ...
    100 * fraction_above_p999 ...
);

fprintf( ...
    "Max / P99:            %.6f\n", ...
    p_max / p99 ...
);

fprintf( ...
    "Max / P99.9:          %.6f\n", ...
    p_max / p999 ...
);

fprintf( ...
    "Std / IQR:            %.6f\n", ...
    p_std / p_iqr ...
);

fprintf("========================================\n");

figure;

semilogy(p_max_t, "DisplayName", "max(p)");
hold on;

semilogy(p_mean_t, "DisplayName", "mean(p)");
semilogy(p_median_t, "DisplayName", "median(p)");

xlabel("Timestep");
ylabel("Pressure");

title("Pressure Statistics Over Time");

legend("Location", "best");

grid on;