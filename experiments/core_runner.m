function results = core_runner(X_train, Y_train, X_test, Y_test, true_labels_test, true_mixture, K, M, options, models_to_run, verbose)
% CORE_RUNNER
% Standardized execution engine for evaluating distributed MoE models on a single dataset.
%
% Inputs:
%   X_train, Y_train       : Training covariates and responses
%   X_test, Y_test         : Testing covariates and responses (unseen data)
%   true_labels_test       : True component labels for test set
%   true_mixture           : Ground truth mixture struct (for KL divergence & MSE)
%   K                      : Number of experts (default 5)
%   M                      : Number of distributed machines
%   options                : Configuration struct (get_options('default'))
%   models_to_run          : (Optional) Cell array of models to run, e.g.:
%                            {'DME', 'MED', 'WAVR', 'AAVR', 'GM', 'FED'}
%                            Default: all available models
%   verbose                : (Optional) 0: silent, 1: print summaries (default 0)
%
% Output:
%   results : Struct containing per-model results:
%             results.(model).param   - Estimated parameters (Beta, Alpha, etc.)
%             results.(model).metrics - 1x9 vector of performance criteria:
%               [Time, Trandis, Loglik, mse_param, RPE, Correlation, RI, ARI, ClustErr]
%             results.(model).fit     - Cleaned model struct (intermediate data stripped)

    if nargin < 10 || isempty(models_to_run)
        models_to_run = {'DME', 'MED', 'WAVR', 'AAVR', 'GM', 'FED'};
    end
    if nargin < 11 || isempty(verbose)
        verbose = 0;
    end

    results = struct();

    % -----------------------------------------------------------------
    % 1. Fit Local Models Once (Shared across one-shot aggregators)
    % -----------------------------------------------------------------
    need_one_shot = any(ismember({'DME', 'MED', 'WAVR', 'AAVR', 'GM'}, models_to_run));
    DMEfit = [];

    if need_one_shot
        if verbose >= 1
            fprintf('    --> [1/2] Fitting M=%d local models sequentially & performing DME aggregation...\n', M);
        end
        % Distributed_MixtureOfExperts_Gaussian fits local models on M machines
        % and performs one-shot optimal-transport aggregation (DME).
        DMEfit = Distributed_MixtureOfExperts_Gaussian(X_train, Y_train, K, M, options);
    end

    % Extract pooled local mixture f^W if available
    f_W = [];
    if need_one_shot && ~isempty(DMEfit) && isfield(DMEfit, 'large_mixture')
        f_W = DMEfit.large_mixture;
    end

    % -----------------------------------------------------------------
    % 1b. Batch FedAvg Trajectory (Optimizes multi-round FedAvg)
    % -----------------------------------------------------------------
    fed_models = models_to_run(startsWith(models_to_run, 'FED_T'));
    fed_snapshots = struct();
    if length(fed_models) > 1
        t_vals = zeros(1, length(fed_models));
        for fi = 1:length(fed_models)
            t_str = strrep(fed_models{fi}, 'FED_T', '');
            t_vals(fi) = str2double(t_str);
        end
        t_vals = sort(t_vals(~isnan(t_vals)));
        fed_opt = options;
        fed_opt.FedAvg_rounds    = max(t_vals);
        fed_opt.FedAvg_snapshots = t_vals;
        fed_traj = FedAvg_MixtureOfExperts(X_train, Y_train, K, M, fed_opt);
        if isfield(fed_traj, 'snapshots')
            fed_snapshots = fed_traj.snapshots;
        end
    end

    % -----------------------------------------------------------------
    % 2. Run Candidate Estimators
    % -----------------------------------------------------------------
    for i = 1:length(models_to_run)
        model = models_to_run{i};
        fit = [];
        if verbose >= 1
            fprintf('    --> [2/2] Running & evaluating estimator [%d/%d]: %s...\n', i, length(models_to_run), model);
        end

        switch model
            case 'DME'
                fit = DMEfit;

            case 'MED'
                fit = Median_MixtureOfExperts(DMEfit, K, M, options);

            case 'WAVR'
                fit = Averaged_MixtureOfExperts(DMEfit, K, M, options);

            case 'AAVR'
                fit = Aligned_MixtureOfExperts(DMEfit, K, M, options);

            case 'GM'
                fit = Greedy_MixtureOfExperts(DMEfit, K, M, options);

            case 'FED'
                fit = FedAvg_MixtureOfExperts(X_train, Y_train, K, M, options);

            case {'FED_T1', 'FED_T5', 'FED_T10', 'FED_T20'}
                if isfield(fed_snapshots, model)
                    fit = fed_snapshots.(model);
                else
                    t_round = str2double(strrep(model, 'FED_T', ''));
                    fed_opt = options;
                    fed_opt.FedAvg_rounds = t_round;
                    fit = FedAvg_MixtureOfExperts(X_train, Y_train, K, M, fed_opt);
                end

            otherwise
                warning('Unknown model: %s. Skipping.', model);
                continue;
        end

        % Evaluate model on unseen test set
        [learning_time, trandis, loglik, mse_param, RPE_test, corr_test, RI_test, ARI_test, ClustErr_test, trandis_fW] = ...
            compute_metrics(fit, true_mixture, X_test, Y_test, true_labels_test, verbose, model, f_W);

        metrics = [learning_time, trandis, loglik, mse_param, RPE_test, corr_test, RI_test, ARI_test, ClustErr_test, trandis_fW];

        % Memory cleanup: strip large validation/training matrices from fit before storage
        if isfield(fit, 'X_val'), fit.X_val = []; end
        if isfield(fit, 'Y_val'), fit.Y_val = []; end
        if isfield(fit, 'large_mixture'), fit.large_mixture = []; end
        if (~isfield(options, 'keep_local_estimates') || ~options.keep_local_estimates) && isfield(fit, 'local_estimates')
            fit.local_estimates = [];
        end

        % Store
        results.(model).param   = fit.param;
        results.(model).metrics = metrics;
        results.(model).fit     = fit;
    end

    results.f_W = f_W;
end
