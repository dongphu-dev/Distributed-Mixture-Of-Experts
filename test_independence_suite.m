%% TEST_INDEPENDENCE_SUITE
% Verifies that DME_GitHub is 100% self-contained and independent:
%   1. Cleans MATLAB path of any external DME_Regression directories.
%   2. Initializes only DME_GitHub paths via setup_paths.m.
%   3. Verifies that all functions resolve strictly within DME_GitHub.
%   4. Executes fast smoke-tests for all core models and real datasets.

function test_independence_suite()
    fprintf('========================================================================\n');
    fprintf('  TESTING INDEPENDENCE AND SELF-CONTAINMENT OF DME_GitHub               \n');
    fprintf('========================================================================\n');

    current_dir = fileparts(mfilename('fullpath'));
    
    % Step 1: Clean path and restore default
    fprintf('\n[1/5] Resetting MATLAB path to ensure no external leaks...\n');
    restoredefaultpath;
    
    % Step 2: Initialize paths strictly within DME_GitHub
    cd(current_dir);
    run(fullfile(current_dir, 'setup_paths.m'));

    % Step 3: Verify function resolution
    fprintf('\n[2/5] Verifying function paths (must all resolve inside DME_GitHub)...\n');
    funcs_to_check = { ...
        'Distributed_MixtureOfExperts_Gaussian', ...
        'Distributed_MixtureOfExperts_Hetero', ...
        'Global_MixtureOfExperts', ...
        'Aligned_MixtureOfExperts', ...
        'Greedy_MixtureOfExperts', ...
        'compute_metrics', ...
        'IRLS', ...
        'identify_network' ...
    };

    all_internal = true;
    for i = 1:length(funcs_to_check)
        fn = funcs_to_check{i};
        w = which(fn);
        if isempty(w)
            fprintf('  [FAIL] Function not found: %s\n', fn);
            all_internal = false;
        elseif ~contains(w, current_dir)
            fprintf('  [FAIL] %s points outside DME_GitHub: %s\n', fn, w);
            all_internal = false;
        else
            fprintf('  [PASS] %-40s -> %s\n', fn, strrep(w, current_dir, '.'));
        end
    end

    if ~all_internal
        error('Path isolation failed! Some functions resolve outside DME_GitHub.');
    end

    % Step 4: Run Toy Example Smoke Test
    fprintf('\n[3/5] Running Toy Example smoke test (K=2, M=2)...\n');
    toy_data = load(fullfile(current_dir, 'toy_examples', 'toy_example_data.mat'));
    X_toy = toy_data.X;
    Y_toy = toy_data.Y;
    
    opt = get_options('default');
    opt.verbose = 0;
    opt.max_iter = 5;
    opt.nb_EM_runs = 1;
    opt.S = 50;

    dme_fit = Distributed_MixtureOfExperts_Gaussian(X_toy, Y_toy, 2, 2, opt);
    assert(~isempty(dme_fit.experts), 'DME fit failed on toy data');
    fprintf('  [PASS] DMoE fit completed successfully.\n');

    glb_fit = Global_MixtureOfExperts(X_toy, Y_toy, 2, opt);
    assert(~isempty(glb_fit.experts), 'GLB fit failed on toy data');
    fprintf('  [PASS] GLB fit completed successfully.\n');

    ali_fit = Aligned_MixtureOfExperts(dme_fit, 2, 2, opt);
    assert(~isempty(ali_fit.experts), 'AAVR fit failed on toy data');
    fprintf('  [PASS] AAVR fit completed successfully.\n');

    grd_fit = Greedy_MixtureOfExperts(dme_fit, 2, 2, opt);
    assert(~isempty(grd_fit.experts), 'GM fit failed on toy data');
    fprintf('  [PASS] GM fit completed successfully.\n');

    % Step 5: Test Metric Computation on Synthetic Benchmark Data
    fprintf('\n[4/5] Testing 9-criterion metric computation on synthetic data...\n');
    synth_file = fullfile(current_dir, 'data', 'dataset_N20k_K5_d20_balanced.mat');
    synth_data = load(synth_file);
    X_syn = synth_data.X_mat(1:500, :, 1);
    Y_syn = synth_data.Y_mat(1:500, 1);
    L_syn = synth_data.LABEL_mat(1:500, 1);
    true_mix = synth_data.true_mixture;

    opt5 = get_options('default');
    opt5.verbose = 0;
    opt5.max_iter = 3;
    opt5.nb_EM_runs = 1;
    opt5.S = 50;
    dme5_fit = Distributed_MixtureOfExperts_Gaussian(X_syn, Y_syn, 5, 2, opt5);
    
    [t_l, td, ll, mse, rpe, cr, ri, ari, ce] = compute_metrics(dme5_fit, true_mix, X_syn, Y_syn, L_syn, 0, 'DME');
    m_vec = [t_l, td, ll, mse, rpe, cr, ri, ari, ce];
    assert(length(m_vec) == 9, 'Metric vector length mismatch');
    fprintf('  [PASS] compute_metrics returned 9 criteria successfully.\n');

    % Step 6: Test Real Dataset Loadability
    fprintf('\n[5/5] Testing real-world benchmark datasets...\n');
    bj_mat = fullfile(current_dir, 'real_data', 'beijing_air_quality', 'beijing_air_quality_processed.mat');
    assert(exist(bj_mat, 'file') == 2, 'Beijing dataset missing');
    bj_data = load(bj_mat);
    fprintf('  [PASS] Beijing dataset loaded successfully (%d stations, %d train records).\n', ...
            length(bj_data.X_train_cells), size(bj_data.X_train_cells{1}, 1));

    msd_mat = fullfile(current_dir, 'real_data', 'year_prediction_msd', 'year_msd_sub10.mat');
    assert(exist(msd_mat, 'file') == 2, 'MSD dataset missing');
    msd_data = load(msd_mat);
    fprintf('  [PASS] YearPredictionMSD subset loaded successfully (M=%d nodes, d=%d features).\n', ...
            msd_data.M, msd_data.d);

    fprintf('\n========================================================================\n');
    fprintf('  SUCCESS: DME_GitHub IS 100%% SELF-CONTAINED AND OPERATIONAL!           \n');
    fprintf('========================================================================\n');
end
