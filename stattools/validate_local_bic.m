function bic_results = validate_local_bic()
% VALIDATE_LOCAL_BIC
% Empirically validates that local machines naturally identify their true local
% expert count K_m using Bayesian Information Criterion (BIC).
%
% Formula:
%   BIC(k) = -2 * loglik + nu_k * log(N_m)
% where nu_k = (k - 1) * (d + 1) + k * (d + 2) is the number of free parameters.
%
% Output:
%   Saved to results/heterogeneous_Km/BIC_Validation_Results.mat

    fprintf('========================================================\n');
    fprintf('  Validating Local Model Selection via BIC             \n');
    fprintf('========================================================\n');

    % Load heterogeneous dataset
    current_dir = fileparts(mfilename('fullpath'));
    base_dir = fullfile(current_dir, '..');
    data_file = fullfile(base_dir, 'data', 'dataset_hetero_Km_K5_d20.mat');

    if ~exist(data_file, 'file')
        error('Dataset %s not found. Run data_generator_hetero_Km first.', data_file);
    end

    data = load(data_file);
    X_cells = data.X_cells;
    Y_cells = data.Y_cells;
    config  = data.config;

    k_candidates = 1:4;
    M = config.M;
    d = config.d;

    options = get_options('default');
    options.nb_EM_runs = 5;
    options.max_iter   = 30;
    options.verbose    = 0;

    bic_matrix = zeros(M, length(k_candidates));
    loglik_matrix = zeros(M, length(k_candidates));
    selected_K = zeros(1, M);

    for m = 1:M
        X_m = X_cells{m};
        Y_m = Y_cells{m};
        N_m = size(X_m, 1);
        true_Km = config.K_vec(m);

        fprintf('\n--- Machine %d (True Subpopulation Km = %d) ---\n', m, true_Km);

        for ik = 1:length(k_candidates)
            k = k_candidates(ik);
            try
                fit_k = MixtureOfExperts(X_m, Y_m, k, options);
                ll = fit_k.stats.ml;
                
                % Degrees of freedom for Gaussian MoE
                nu_k = (k - 1) * (d + 1) + k * (d + 2);
                bic_val = -2 * ll + nu_k * log(N_m);

                bic_matrix(m, ik)    = bic_val;
                loglik_matrix(m, ik) = ll;

                fprintf('  k = %d: Loglik = %11.2f, DoF = %3d, BIC = %12.2f\n', ...
                        k, ll, nu_k, bic_val);
            catch ME
                fprintf('  k = %d: Fitting failed (%s)\n', k, ME.message);
                bic_matrix(m, ik) = inf;
            end
        end

        [min_bic, best_ik] = min(bic_matrix(m, :));
        selected_K(m) = k_candidates(best_ik);
        fprintf('  => Optimal by BIC: K_%d = %d (True Km = %d, Match: %s)\n', ...
                m, selected_K(m), true_Km, string(selected_K(m) == true_Km));
    end

    % Save results
    res_dir = fullfile(base_dir, 'results', 'heterogeneous_Km');
    if ~exist(res_dir, 'dir'), mkdir(res_dir); end

    bic_results.bic_matrix    = bic_matrix;
    bic_results.loglik_matrix = loglik_matrix;
    bic_results.k_candidates  = k_candidates;
    bic_results.selected_K    = selected_K;
    bic_results.true_K_vec    = config.K_vec;

    save(fullfile(res_dir, 'BIC_Validation_Results.mat'), 'bic_results');
    fprintf('\nBIC Validation results saved to: %s\n', fullfile(res_dir, 'BIC_Validation_Results.mat'));

end
