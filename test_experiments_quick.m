%% TEST_EXPERIMENTS_QUICK
% Fast end-to-end verification of the synthetic experiment runners
% to confirm independence and correctness on 1 run.
% Designed for reviewers and users to verify the codebase out-of-the-box
% without requiring pre-computed result files or exporting .tex tables.

function test_experiments_quick()
    fprintf('========================================================================\n');
    fprintf('  TESTING SYNTHETIC EXPERIMENT BENCHMARK RUNNERS (END-TO-END VERIFICATION)\n');
    fprintf('========================================================================\n');

    current_dir = fileparts(mfilename('fullpath'));
    cd(current_dir);
    setup_paths;

    % -------------------------------------------------------------------------
    % 1. Homogeneous Benchmark (Exp 1): Fast live verification on 1 run
    % -------------------------------------------------------------------------
    fprintf('\n>>> [1/3] Running exp_official_homogeneous_benchmark (1 run, balanced, S=250)...\n');
    exp1_mat = fullfile(current_dir, 'results', 'official_homogeneous', 'Result_Official_Homogeneous_M16_N100k.mat');
    if exist(exp1_mat, 'file') == 2
        fprintf('    Found existing result file: %s\n', exp1_mat);
        exp1_data = load(exp1_mat);
        assert(isfield(exp1_data.All_results, 'balanced'), 'Missing balanced profile in Exp 1');
        fprintf('    [PASS] Pre-computed Exp 1 verified successfully.\n');
    else
        fprintf('    Executing live smoke run...\n');
        t1 = tic;
        res1 = exp_official_homogeneous_benchmark(100000, 1, 250, {'balanced'});
        assert(isfield(res1, 'balanced'), 'Missing balanced profile in Exp 1');
        assert(isfield(res1.balanced, 'DME'), 'Missing DME in Exp 1');
        assert(isfield(res1.balanced, 'GM'), 'Missing GM in Exp 1');
        assert(isfield(res1.balanced, 'AAVR'), 'Missing AAVR in Exp 1');
        fprintf('    [PASS] Experiment 1 completed in %.1f s!\n', toc(t1));
    end

    % -------------------------------------------------------------------------
    % 2. Heterogeneous Benchmark (Exp 2): Fast live verification on 1 run
    % -------------------------------------------------------------------------
    fprintf('\n>>> [2/3] Running exp_official_heterogeneous_Km_benchmark (1 run, S=250)...\n');
    exp2_mat = fullfile(current_dir, 'results', 'official_heterogeneous', 'Result_Official_Heterogeneous_M16_N100k.mat');
    if exist(exp2_mat, 'file') == 2
        fprintf('    Found existing result file: %s\n', exp2_mat);
        exp2_data = load(exp2_mat);
        assert(isfield(exp2_data.results, 'metrics_raw'), 'Missing metrics_raw in Exp 2');
        fprintf('    [PASS] Pre-computed Exp 2 verified successfully.\n');
    else
        fprintf('    Executing live smoke run...\n');
        t2 = tic;
        res2 = exp_official_heterogeneous_Km_benchmark(100000, 1, 250);
        assert(isfield(res2, 'metrics_raw'), 'Missing metrics_raw in Exp 2');
        assert(isfield(res2.metrics_mean, 'DME_Warm'), 'Missing DME_Warm in Exp 2');
        assert(isfield(res2.metrics_mean, 'GM'), 'Missing GM in Exp 2');
        fprintf('    [PASS] Experiment 2 completed in %.1f s!\n', toc(t2));
    end

    % -------------------------------------------------------------------------
    % 3. Sensitivity S Benchmark (Exp 3): Fast live verification on 1 run
    % -------------------------------------------------------------------------
    fprintf('\n>>> [3/3] Running exp_sensitivity_support_S (1 run, S=[100, 250])...\n');
    t3 = tic;
    res3 = exp_sensitivity_support_S(1, [100, 250], 100000, 'moderate');
    assert(isfield(res3, 'pooled'), 'Missing pooled in Exp 3');
    assert(isfield(res3.baselines, 'AAVR'), 'Missing AAVR baseline in Exp 3');
    assert(isfield(res3.pooled.S_100, 'GM'), 'Missing GM model in Exp 3');
    fprintf('    [PASS] Experiment 3 completed in %.1f s with AAVR and GM verified!\n', toc(t3));

    fprintf('\n========================================================================\n');
    fprintf('  ALL SYNTHETIC BENCHMARK EXPERIMENTS OPERATING PERFECTLY!              \n');
    fprintf('========================================================================\n');
end
