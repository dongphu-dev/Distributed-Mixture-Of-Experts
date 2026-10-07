%% Test script for YearPredictionMSD on 1/10 subsample (M = 32)
cd(fileparts(mfilename('fullpath')));
fprintf('Testing YearPredictionMSD benchmark on 1/10 data with M=32 workers...\n');
use_sub10  = true;
K          = 4;
S          = 1000;
max_iter   = 20;
nb_EM_runs = 1;
res = run_year_msd_benchmark(use_sub10, K, S, max_iter, nb_EM_runs);
fprintf('\n==> YearPredictionMSD 1/10 test finished successfully!\n');
