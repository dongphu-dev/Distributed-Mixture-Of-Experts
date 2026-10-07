%% RUN_TEST_K4
% Run Beijing Air Quality benchmark with production settings: K=4, S=2000, max_iter=20, nb_EM_runs=2
cd(fileparts(mfilename('fullpath')));
fprintf('Starting Beijing Benchmark with K=4, S=2000...\n');
res = run_beijing_benchmark(4, 2000, 20, 2);
fprintf('Completed successfully!\n');
