
function [n_iter, info] = get_nonlinear_iterations(log_file)
% ============================================================
% GET NONLINEAR ITERATIONS
%
% Reads the LAST nonlinear-solver iteration table contained
% in a COMSOL progress log and returns the number of nonlinear
% iterations performed in that solve.
%
% The function:
%   1. Finds all "Nonlinear solver" sections.
%   2. Selects the last one.
%   3. Finds the corresponding iteration-table header.
%   4. Reads only consecutive iteration rows.
%   5. Verifies that the iteration numbering is consistent.
%
% INPUT
%   log_file : path to the COMSOL solver log
%
% OUTPUT
%   n_iter   : number of nonlinear iterations
%
% ============================================================


%% Check input file

if ~isfile(log_file)
    error('COMSOL solver log not found: %s', log_file);
end


%% Read log

log_text = fileread(log_file);

if isempty(log_text)
    error('COMSOL solver log is empty: %s', log_file);
end


%% Split into lines

lines = splitlines(log_text);


%% Find all "Nonlinear solver" sections

nonlinear_idx = find(contains(lines, 'Nonlinear solver'));

if isempty(nonlinear_idx)
    error('No "Nonlinear solver" section found in COMSOL log.');
end


%% Select the last nonlinear-solver section

start_line = nonlinear_idx(end);


%% Define the end of the section
%
% Normally this is the end of the file, but keeping an explicit
% section boundary prevents parsing unrelated information.

if length(nonlinear_idx) > 1
    % We already selected the last section, so its natural end
    % is the end of the log.
    end_line = length(lines);
else
    end_line = length(lines);
end


%% Find iteration-table header
%
% We require the first non-space characters to be "Iter".
% This is stricter than simply searching for the word anywhere
% in the line.

header_line = [];

for i = start_line:end_line

    line = strtrim(lines{i});

    if ~isempty(regexp(line, '^Iter(\s|$)', 'once'))
        header_line = i;
        break;
    end

end

if isempty(header_line)
    error(['Nonlinear iteration table header not found after ' ...
           'the last "Nonlinear solver" section.']);
end


%% Read consecutive iteration rows

iteration_numbers = [];
iteration_lines = strings(0,1);

started = false;

for i = header_line + 1:end_line

    line = strtrim(lines{i});

    % Empty line:
    % ignore it before the table starts,
    % terminate parsing after iteration rows have started.
    if isempty(line)

        if started
            break;
        else
            continue;
        end

    end


    % A valid iteration row must start with an integer followed
    % by whitespace.
    %
    % Examples:
    %
    %   1    0.0123    ...
    %   2    0.0014    ...
    %
    % but not:
    %
    %   Time = 0.01
    %   Solution 1

    token = regexp(line, '^(\d+)\s+', 'tokens', 'once');

    if ~isempty(token)

        iteration_number = str2double(token{1});

        if ~isfinite(iteration_number)
            error('Invalid nonlinear iteration number in log.');
        end

        iteration_numbers(end+1,1) = iteration_number;
        iteration_lines(end+1,1) = string(line);
        
        started = true;

    else

        % Once the iteration table has started, the first line
        % that does not have the expected format terminates it.
        if started
            break;
        end

    end

end


%% Validate extracted iterations

if isempty(iteration_numbers)
    error('No nonlinear iteration rows found in COMSOL log.');
end


% Iteration numbers should increase monotonically.
if any(diff(iteration_numbers) <= 0)
    error(['Nonlinear iteration numbers are not strictly increasing. ' ...
           'The COMSOL log may have been parsed incorrectly.']);
end


% Iteration numbering should normally be consecutive.
if length(iteration_numbers) > 1

    if any(diff(iteration_numbers) ~= 1)
        error(['Nonlinear iteration numbering is not consecutive. ' ...
               'The COMSOL log may contain an unexpected format.']);
    end

end


%% Return number of nonlinear iterations
%
% We use the number of actual iteration rows rather than simply
% max(iteration_numbers). This makes explicit what we are
% measuring: the number of nonlinear iteration records present
% in the COMSOL table.

n_iter = length(iteration_numbers);

%% Parse nonlinear solver table

num_rows = length(iteration_lines);

table_data = nan(num_rows, 10);

for j = 1:num_rows

    values = sscanf(char(iteration_lines(j)), '%f');

    if length(values) ~= 10
        error( ...
            ['Unexpected COMSOL nonlinear solver row format.\n' ...
             'Expected 10 numeric columns, found %d.\n' ...
             'Row: %s'], ...
             length(values), ...
             iteration_lines(j));
    end

    table_data(j,:) = values(:).';

end

%% Verify parsed iteration column

if ~isequal(table_data(:,1), iteration_numbers)
    error( ...
        ['Parsed Iter column does not match the iteration numbers ' ...
         'identified from the COMSOL log.']);
end

%% Build diagnostic information

info = struct();

info.log_file = log_file;
info.section_start_line = start_line;
info.header_line = header_line;
info.header = strtrim(lines{header_line});

info.iteration_numbers = iteration_numbers;
info.iteration_lines = iteration_lines;

info.first_iteration = iteration_numbers(1);
info.last_iteration = iteration_numbers(end);
info.number_of_rows = length(iteration_numbers);

%% Store parsed nonlinear solver data

info.iteration = table_data(:,1);
info.sol_est   = table_data(:,2);
info.res_est   = table_data(:,3);
info.damping   = table_data(:,4);
info.stepsize  = table_data(:,5);

info.num_res = table_data(:,6);
info.num_jac = table_data(:,7);
info.num_sol = table_data(:,8);

info.lin_err = table_data(:,9);
info.lin_res = table_data(:,10);

%% Store final cumulative solver statistics

% #Res, #Jac and #Sol are cumulative counters in the COMSOL log.
% Therefore, the last row contains the total number performed
% during the nonlinear solve.

info.total_residual_evaluations = info.num_res(end);
info.total_jacobian_evaluations = info.num_jac(end);
info.total_linear_solves        = info.num_sol(end);

%% Additional consistency check

expected_last_iteration = ...
    iteration_numbers(1) + n_iter - 1;

if iteration_numbers(end) ~= expected_last_iteration

    error( ...
        ['Internal consistency check failed. ' ...
         'First iteration: %d, last iteration: %d, rows found: %d.'], ...
         iteration_numbers(1), ...
         iteration_numbers(end), ...
         n_iter);

end

end

