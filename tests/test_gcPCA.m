% Tests for generalized contrastive PCA (MATLAB implementation)
%
% Run from repository root:
%   matlab -batch "cd tests; test_gcPCA"

% Add matlab source to path
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'matlab'));

test_count = 0;
pass_count = 0;
fail_count = 0;
failures = {};

%% Generate synthetic data
rng(42);
n_a = 80;
n_b = 60;
p = 20;

Rb = randn(n_b, p);
Ra = randn(n_a, p);
Ra(:, 1) = Ra(:, 1) + randn(n_a, 1) * 5;
Ra(:, 2) = Ra(:, 2) + randn(n_a, 1) * 3;

% Small data for quick tests
rng(42);
n_a_sm = 30;
n_b_sm = 25;
p_sm = 8;
Rb_sm = randn(n_b_sm, p_sm);
Ra_sm = randn(n_a_sm, p_sm);
Ra_sm(:, 1) = Ra_sm(:, 1) + randn(n_a_sm, 1) * 4;

fprintf('\n========================================\n');
fprintf('  MATLAB gcPCA Tests\n');
fprintf('========================================\n\n');

%% Test gcPCA: non-orthogonal methods
fprintf('--- gcPCA non-orthogonal methods ---\n');

for ver = [2, 3, 4]
    try
        [B, S, X] = gcPCA(Ra, Rb, ver);
        [test_count, pass_count] = check(true, sprintf('gcPCA v%d fits without error', ver), test_count, pass_count);

        [test_count, pass_count] = check(size(X, 1) == p, ...
            sprintf('gcPCA v%d: loadings has %d rows (features)', ver, p), test_count, pass_count);
        [test_count, pass_count] = check(~isempty(B.a), ...
            sprintf('gcPCA v%d: B.a exists', ver), test_count, pass_count);
        [test_count, pass_count] = check(~isempty(B.b), ...
            sprintf('gcPCA v%d: B.b exists', ver), test_count, pass_count);
        [test_count, pass_count] = check(~isempty(S.objval), ...
            sprintf('gcPCA v%d: S.objval exists', ver), test_count, pass_count);
    catch ME
        test_count = test_count + 1;
        fail_count = fail_count + 1;
        failures{end+1} = sprintf('gcPCA v%d: %s', ver, ME.message);
        fprintf('  FAIL: gcPCA v%d: %s\n', ver, ME.message);
    end
end

%% Test gcPCA v1 (contrastive PCA)
fprintf('\n--- gcPCA v1 (contrastive PCA) ---\n');
try
    [B, S, X] = gcPCA(Ra, Rb, 1);
    [test_count, pass_count] = check(true, 'gcPCA v1 fits without error', test_count, pass_count);
    [test_count, pass_count] = check(size(X, 1) == p, ...
        'gcPCA v1: loadings has correct rows', test_count, pass_count);
catch ME
    test_count = test_count + 1;
    fail_count = fail_count + 1;
    failures{end+1} = sprintf('gcPCA v1: %s', ME.message);
    fprintf('  FAIL: gcPCA v1: %s\n', ME.message);
end

%% Test gcPCA: orthogonal methods
fprintf('\n--- gcPCA orthogonal methods ---\n');

for ver = [2.1, 3.1, 4.1]
    try
        [B, S, X] = gcPCA(Ra_sm, Rb_sm, ver, 'Ncalc', 4);
        [test_count, pass_count] = check(true, sprintf('gcPCA v%.1f fits without error', ver), test_count, pass_count);

        gram = X' * X;
        off_diag = gram - diag(diag(gram));
        off_diag_max = max(abs(off_diag(:)));
        [test_count, pass_count] = check(off_diag_max < 1e-5, ...
            sprintf('gcPCA v%.1f: loadings orthogonal (max off-diag = %.2e)', ver, off_diag_max), test_count, pass_count);
    catch ME
        test_count = test_count + 1;
        fail_count = fail_count + 1;
        failures{end+1} = sprintf('gcPCA v%.1f: %s', ver, ME.message);
        fprintf('  FAIL: gcPCA v%.1f: %s\n', ver, ME.message);
    end
end

%% Test gcPCA: v4 objective values bounded
fprintf('\n--- gcPCA v4 value bounds ---\n');
[~, S, ~] = gcPCA(Ra, Rb, 4);
objval = S.objval;
[test_count, pass_count] = check(all(objval >= -1 - 1e-10), ...
    'v4 objective values >= -1', test_count, pass_count);
[test_count, pass_count] = check(all(objval <= 1 + 1e-10), ...
    'v4 objective values <= 1', test_count, pass_count);

%% Test gcPCA: v2 objective values positive
fprintf('\n--- gcPCA v2 value bounds ---\n');
[~, S, ~] = gcPCA(Ra, Rb, 2);
[test_count, pass_count] = check(all(S.objval > 0), ...
    'v2 objective values > 0', test_count, pass_count);

%% Test gcPCA: different feature counts should error
fprintf('\n--- gcPCA input validation ---\n');
try
    gcPCA(randn(10, 5), randn(10, 6), 4);
    test_count = test_count + 1;
    fail_count = fail_count + 1;
    failures{end+1} = 'Different feature counts should error';
    fprintf('  FAIL: Different feature counts should error\n');
catch
    [test_count, pass_count] = check(true, 'Different feature counts raises error', test_count, pass_count);
end

%% Test gcPCA: rank-deficient data
fprintf('\n--- gcPCA rank-deficient data ---\n');
rng(42);
Ra_rd = randn(5, 20);
Rb_rd = randn(5, 20);
try
    [~, ~, X_rd] = gcPCA(Ra_rd, Rb_rd, 4);
    [test_count, pass_count] = check(~isempty(X_rd), 'Rank-deficient data handled', test_count, pass_count);
catch ME
    test_count = test_count + 1;
    fail_count = fail_count + 1;
    failures{end+1} = sprintf('Rank-deficient: %s', ME.message);
    fprintf('  FAIL: Rank-deficient: %s\n', ME.message);
end

%% Test gcPCA: equal data gives zero v4 values
fprintf('\n--- gcPCA equal data ---\n');
rng(42);
data_eq = randn(30, 10);
[~, S_eq, ~] = gcPCA(data_eq, data_eq, 4, 'normalize', false);
objval_eq = S_eq.objval;
[test_count, pass_count] = check(all(abs(objval_eq) < 1e-10), ...
    'Equal data gives zero v4 objective values', test_count, pass_count);

%% Test gcPCA: reproducibility
fprintf('\n--- gcPCA reproducibility ---\n');
[~, ~, X1] = gcPCA(Ra, Rb, 4);
[~, ~, X2] = gcPCA(Ra, Rb, 4);
[test_count, pass_count] = check(max(abs(abs(X1(:)) - abs(X2(:)))) < 1e-10, ...
    'Same data gives same loadings', test_count, pass_count);

%% Test sparse_gcPCA: basic fit for v2-v4
fprintf('\n--- sparse_gcPCA ---\n');

lambdas = [0.1, 0.5, 1.0];

for ver = [2, 3, 4]
    try
        [B, S, X] = sparse_gcPCA(Ra_sm, Rb_sm, ver, 'Nsparse', 2, 'lasso_penalty', lambdas);
        [test_count, pass_count] = check(true, sprintf('sparse_gcPCA v%d fits without error', ver), test_count, pass_count);
        [test_count, pass_count] = check(length(X) == length(lambdas), ...
            sprintf('sparse_gcPCA v%d: correct number of loading sets', ver), test_count, pass_count);
        [test_count, pass_count] = check(size(X{1}, 1) == p_sm, ...
            sprintf('sparse_gcPCA v%d: loadings have correct rows', ver), test_count, pass_count);
        [test_count, pass_count] = check(size(X{1}, 2) == 2, ...
            sprintf('sparse_gcPCA v%d: loadings have Nsparse columns', ver), test_count, pass_count);
    catch ME
        test_count = test_count + 1;
        fail_count = fail_count + 1;
        failures{end+1} = sprintf('sparse_gcPCA v%d: %s', ver, ME.message);
        fprintf('  FAIL: sparse_gcPCA v%d: %s\n', ver, ME.message);
    end
end

%% Test sparse_gcPCA: scores output
fprintf('\n--- sparse_gcPCA scores ---\n');
[B, S, X] = sparse_gcPCA(Ra_sm, Rb_sm, 4, 'Nsparse', 2, 'lasso_penalty', lambdas);
[test_count, pass_count] = check(length(B.a) == length(lambdas), ...
    'B.a cell length matches lambdas', test_count, pass_count);
[test_count, pass_count] = check(length(B.b) == length(lambdas), ...
    'B.b cell length matches lambdas', test_count, pass_count);
[test_count, pass_count] = check(length(S.a) == length(lambdas), ...
    'S.a cell length matches lambdas', test_count, pass_count);
[test_count, pass_count] = check(length(S.b) == length(lambdas), ...
    'S.b cell length matches lambdas', test_count, pass_count);

%% Test sparse_gcPCA v1
fprintf('\n--- sparse_gcPCA v1 ---\n');
try
    [B, S, X] = sparse_gcPCA(Ra_sm, Rb_sm, 1, 'Nsparse', 2, 'lasso_penalty', lambdas);
    [test_count, pass_count] = check(true, 'sparse_gcPCA v1 fits without error', test_count, pass_count);
catch ME
    test_count = test_count + 1;
    fail_count = fail_count + 1;
    failures{end+1} = sprintf('sparse_gcPCA v1: %s', ME.message);
    fprintf('  FAIL: sparse_gcPCA v1: %s\n', ME.message);
end

%% Summary
fprintf('\n========================================\n');
fprintf('  Results: %d passed, %d failed out of %d tests\n', pass_count, fail_count, test_count);
fprintf('========================================\n');

if fail_count > 0
    fprintf('\nFailed tests:\n');
    for i = 1:length(failures)
        fprintf('  - %s\n', failures{i});
    end
    exit(1);
else
    fprintf('\nAll tests passed!\n');
    exit(0);
end

%% Helper function
function [tc, pc] = check(condition, msg, tc, pc)
    tc = tc + 1;
    if condition
        pc = pc + 1;
        fprintf('  PASS: %s\n', msg);
    else
        fprintf('  FAIL: %s\n', msg);
    end
end
