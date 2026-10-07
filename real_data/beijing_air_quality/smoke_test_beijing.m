%% SMOKE_TEST_BEIJING
% Quick verification with K=2, max_iter=3, S=100
cd(fileparts(mfilename('fullpath')));
fprintf('Starting Beijing smoke test...\n');
res = run_beijing_benchmark(2, 100, 3, 1);
fprintf('Smoke test completed successfully!\n');
