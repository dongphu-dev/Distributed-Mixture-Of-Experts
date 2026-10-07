%% Official Production Benchmark on Full YearPredictionMSD Dataset
cd(fileparts(mfilename('fullpath')));
fprintf('========================================================================\n');
fprintf('  STARTING OFFICIAL FULL-SCALE BENCHMARK ON YEARPREDICTIONMSD            \n');
fprintf('  Train N = 463,715 | Test N = 51,630 | d = 90 | M = 32 | K = 4         \n');
fprintf('========================================================================\n');

use_sub10  = false; % Full dataset
K          = 4;
S          = 2000;
max_iter   = 30;
nb_EM_runs = 2;

t_total = tic;
results = run_year_msd_benchmark(use_sub10, K, S, max_iter, nb_EM_runs);
fprintf('\n==> FULL-SCALE YEARPREDICTIONMSD BENCHMARK COMPLETED in %.2f s!\n', toc(t_total));
