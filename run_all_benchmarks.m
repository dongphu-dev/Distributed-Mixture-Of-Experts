%% RUN_ALL_BENCHMARKS
% Master execution script to reproduce the experiments from:
% "Optimal Transport Aggregation for Distributed Mixture-of-Experts" (DMoE)
%
% Usage:
%   run_all_benchmarks;           % Interactive menu / quick run
%   run_all_benchmarks('quick');   % Fast verification mode (DEV)
%   run_all_benchmarks('full');    % Full Monte Carlo reproduction (50 runs)

function run_all_benchmarks(mode)
    if nargin < 1 || isempty(mode)
        mode = 'quick';
    end

    % 1. Initialize paths
    setup_paths;

    fprintf('\n========================================================================\n');
    fprintf('  DISTRIBUTED MIXTURE-OF-EXPERTS (DMoE) BENCHMARK SUITE                 \n');
    fprintf('  Mode: %s                                                              \n', upper(mode));
    fprintf('========================================================================\n');

    if strcmpi(mode, 'quick')
        num_runs = 2;
        N_total  = 100000;
        S        = 2000;
        fprintf('  Running in QUICK mode (2 Monte Carlo replicates for fast verification)\n');
    else
        num_runs = 50;
        N_total  = 100000;
        S        = 2000;
        fprintf('  Running in FULL mode (50 Monte Carlo replicates for paper reproduction)\n');
    end

    % 2. Check / Pre-generate synthetic benchmark datasets
    data_dir = fullfile(fileparts(mfilename('fullpath')), 'data');
    homo_file   = fullfile(data_dir, 'dataset_N100k_K5_d20_balanced.mat');
    hetero_file = fullfile(data_dir, 'dataset_N100k_K5_d20_hetero_M16_Km3.mat');

    if ~exist(homo_file, 'file')
        fprintf('\n===> Pre-generating homogeneous benchmark datasets...\n');
        data_generator_profiles(num_runs, N_total);
    end

    if ~exist(hetero_file, 'file')
        fprintf('\n===> Pre-generating heterogeneous benchmark datasets...\n');
        data_generator_official_heterogeneous(num_runs, N_total);
    end

    % 3. Run Synthetic Benchmarks (Experiments 1, 2, and 3)
    fprintf('\n>>> Executing Synthetic Benchmark Suite (TN1, TN2, Sens S)...\n');
    run_official_experiments(N_total, num_runs, S);

    fprintf('\n========================================================================\n');
    fprintf('  ALL BENCHMARKS COMPLETED SUCCESSFULLY!                                \n');
    fprintf('  Saved result MAT files can be found in results/                       \n');
    fprintf('========================================================================\n');
end
