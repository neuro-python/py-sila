% Generate MATLAB reference outputs for validation
% This script should be run in MATLAB with the SILA toolbox on the path

clear all;
close all;

fprintf('Generating MATLAB reference outputs for SILA validation\n');
fprintf('=========================================================\n\n');

% Add SILA toolbox to path
addpath('../../matlab/SILA-AD-Biomarker-main');

% Load data
fprintf('Loading input data...\n');
data_path = '../../data/sila_input_amyloid.csv';

if ~exist(data_path, 'file')
    error('Input data file not found: %s', data_path);
end

data = readtable(data_path);
fprintf('  Data shape: %d rows x %d columns\n', height(data), width(data));
fprintf('  Columns: %s\n', strjoin(data.Properties.VariableNames, ', '));

% Extract variables
age = data.AGE;
value = data.AMY_GLOBAL_COMPOSITE;
subid = data.RID;

% SILA parameters (matching sila_demo.m)
dt = 0.25;
val0 = 0.79;
maxi = 200;

% Run SILA
fprintf('\nRunning MATLAB SILA...\n');
tic;
[tsila, tdrs] = SILA(age, value, subid, dt, val0, maxi);
elapsed = toc;

fprintf('  tsila: %d rows x %d columns\n', height(tsila), width(tsila));
fprintf('  tdrs: %d rows x %d columns\n', height(tdrs), width(tdrs));
fprintf('  Elapsed time: %.2f seconds\n', elapsed);

% Run SILA_estimate
fprintf('\nRunning MATLAB SILA_estimate...\n');
tic;
test = SILA_estimate(tsila, age, value, subid);
elapsed = toc;

fprintf('  estimates: %d rows x %d columns\n', height(test), width(test));
fprintf('  Elapsed time: %.2f seconds\n', elapsed);

% Save outputs
fprintf('\nSaving MATLAB reference outputs...\n');
writetable(tsila, 'matlab_tsila.csv');
writetable(tdrs, 'matlab_tdrs.csv');
writetable(test, 'matlab_estimates.csv');

fprintf('  matlab_tsila.csv\n');
fprintf('  matlab_tdrs.csv\n');
fprintf('  matlab_estimates.csv\n');

fprintf('\nMAT LAB reference generation complete!\n');
fprintf('Run Python validation script to compare results.\n\n');
