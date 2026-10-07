function run_official_experiments(N_total, num_runs, S)
%% RUN_OFFICIAL_EXPERIMENTS
% Master runner for the two official benchmark experiments:
%   1. exp_official_homogeneous_benchmark (M=16, Homogeneous K_m = 5)
%   2. exp_official_heterogeneous_Km_benchmark (M=16, Heterogeneous K_m = 3, L=48 -> K=5)
%
% Usage in MATLAB Command Window:
%   run_official_experiments;                    % Default: N = 100k, S = 2000
%   run_official_experiments(1000000);           % Scaled to 1M samples
%   run_official_experiments(N_total, num_runs, S);

    if nargin < 3 || isempty(S),       S = 2000; end
    if nargin < 2 || isempty(num_runs)
        if ismac
            num_runs = 3;  % DEV smoke test on macOS
        else
            num_runs = 50; % 50 runs on Linux/VPS
        end
    end
    if nargin < 1 || isempty(N_total), N_total = 100000; end

    fprintf('========================================================================\n');
    fprintf('  STARTING OFFICIAL DME BENCHMARK SUITE (Experiments 1, 2, and Sens S)  \n');
    fprintf('  Configuration: N = %s, S = %d, Runs = %d                              \n', ...
            format_N_str(N_total), S, num_runs);
    fprintf('========================================================================\n');

    base_dir = fileparts(fileparts(mfilename('fullpath')));
    addpath(base_dir);
    addpath(fullfile(base_dir, 'experiments'));
    addpath(fullfile(base_dir, 'models'));
    addpath(fullfile(base_dir, 'datatools'));
    addpath(fullfile(base_dir, 'evaltools'));
    addpath(fullfile(base_dir, 'reporting'));

    % Ensure parallel pool (preserves existing pool if already open)
    poolobj = gcp('nocreate');
    if isempty(poolobj)
        if ismac
            fprintf('Initializing parallel pool with 4 workers on macOS...\n');
            parpool('Processes', 4);
        else
            fprintf('Initializing parallel pool on Linux/VPS (using available cores)...\n');
            parpool;
        end
    else
        fprintf('Using existing parallel pool with %d workers.\n', poolobj.NumWorkers);
    end

    % -------------------------------------------------------------------------
    % 1. Execute Official Heterogeneous Benchmark (TN2)
    % -------------------------------------------------------------------------
    fprintf('\n>>> [1/3] RUNNING OFFICIAL EXPERIMENT 2: Heterogeneous Km Benchmark...\n');
    t_hetero = tic;
    exp_official_heterogeneous_Km_benchmark(N_total, num_runs, S);
    fprintf('\n>>> Official Heterogeneous Benchmark finished in %.1f seconds.\n', toc(t_hetero));

    % -------------------------------------------------------------------------
    % 2. Execute Official Homogeneous Benchmark (TN1)
    % -------------------------------------------------------------------------
    fprintf('\n>>> [2/3] RUNNING OFFICIAL EXPERIMENT 1: Homogeneous Benchmark...\n');
    t_homo = tic;
    exp_official_homogeneous_benchmark(N_total, num_runs, S);
    fprintf('\n>>> Official Homogeneous Benchmark finished in %.1f seconds.\n', toc(t_homo));

    % -------------------------------------------------------------------------
    % 3. Execute Support Sample Size S Sensitivity Benchmark (Sens S)
    % -------------------------------------------------------------------------
    fprintf('\n>>> [3/3] RUNNING SENSITIVITY BENCHMARK (Support Size S)...\n');
    t_sens = tic;
    exp_sensitivity_support_S(num_runs, [], N_total, 'moderate');
    fprintf('\n>>> Sensitivity S Benchmark finished in %.1f seconds.\n', toc(t_sens));

    fprintf('\n========================================================================\n');
    fprintf('  ALL 3 BENCHMARKS (EXP 1, EXP 2, SENS S) COMPLETED SUCCESSFULLY!       \n');
    fprintf('  Results saved to results/:                                            \n');
    fprintf('    - results/official_homogeneous/                                     \n');
    fprintf('    - results/official_heterogeneous/                                   \n');
    fprintf('    - results/sensitivity_support_S/                                    \n');
    fprintf('========================================================================\n');
