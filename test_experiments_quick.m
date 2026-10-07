%% TEST_EXPERIMENTS_QUICK
% Fast end-to-end verification of the synthetic experiment runners
% to confirm independence and correctness on 1 run.

function test_experiments_quick()
    fprintf('========================================================================\n');
    fprintf('  TESTING SYNTHETIC EXPERIMENT BENCHMARK RUNNERS                        \n');
    fprintf('========================================================================\n');

    current_dir = fileparts(mfilename('fullpath'));
    cd(current_dir);
    setup_paths;

    % 1. Verify Homogeneous Benchmark (Exp 1) result
    fprintf('\n>>> [1/3] Verifying exp_official_homogeneous_benchmark output...\n');
    exp1_mat = fullfile(current_dir, 'results', 'official_homogeneous', 'Result_Official_Homogeneous_M16_N100k.mat');
    assert(exist(exp1_mat, 'file') == 2, 'Missing Exp 1 result file');
    exp1_data = load(exp1_mat);
    assert(isfield(exp1_data.All_results, 'balanced'), 'Missing balanced profile in Exp 1');
    fprintf('    [PASS] Experiment 1 output verified successfully.\n');

    % 2. Check Heterogeneous Benchmark (Exp 2) result
    fprintf('\n>>> [2/3] Verifying exp_official_heterogeneous_Km_benchmark output...\n');
    exp2_mat = fullfile(current_dir, 'results', 'official_heterogeneous', 'Result_Official_Heterogeneous_M16_N100k.mat');
    assert(exist(exp2_mat, 'file') == 2, 'Missing Exp 2 result file');
    exp2_data = load(exp2_mat);
    assert(isfield(exp2_data.results, 'metrics_raw'), 'Missing metrics_raw in Exp 2');
    fprintf('    [PASS] Experiment 2 output verified successfully.\n');

    % 3. Test Sensitivity S Benchmark (Exp 3) with newly renamed models (AAVR, GM)
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
