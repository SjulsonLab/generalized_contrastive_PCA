function [B, S, X, info] = spPCA_WminusSoverS(Ns, Nw, Nshuffle)

% function [B, S, X, info] = spPCA_WminusSoverS(Ns, Nw, Nshuffle)
%
% This function does sleep-preserved PCA (spPCA) on binned spike trains.
%
% INPUTS
% Ns        -- (t1 x n) matrix of binned spike trains during sleep
% Nw        -- (t2 x n) matrix of binned spike trains during awake behavior
% Nshuffle  -- (optional) number of times to shuffle for null distribution
%
% OUTPUTS
% B        -- maxSPV scores (temporal eigenvectors)
% S        -- amount of variance captured for sleep, awake, sleep_shuf, awake_shuf
% X        -- maxSPV loadings for both awake and asleep data
% info     -- extra info
%
% Note: Ns and Nw should both be normalized (as in normalize(zscore(Ns), 'norm'))
%
% Analogous to SVD, which decomposes data matrix N into (U * S * T'), this
% function decomposes N into (B * S * X') where B is the temporal
% eigenvectors (or PC scores), S is a diagonal matrix representing amount
% of variance captured, and X is the PC loadings across cells. Instead of
% maximizing var(N) in each dimension, this spPCA finds the spPCs X that maximize
%
%            (x' * (Nw'*Nw - Ns'*Ns) * x) / (x' * Ns'*Ns * x)
%
% where Ns is asleep data and Nw is awake data. Because Ns and Nw may not
% be full rank, X is calculated in a subspace spanned by PCs accounting for
% 99.5% of the variance.
%
% Luke Sjulson, 2021-09-10


% %% for testing
% clear all
% close all
% clc
% 
% % for testing
% load ../data_deficient.mat
% Nw = Nw_tr;
% 
% % Ns(:, 3) = zeros(size(Ns, 1), 1);
% % Ns(:, 4) = zeros(size(Ns, 1), 1);
% % Ns(:, 5) = zeros(size(Ns, 1), 1);
% nargin = 3;
% Nshuffle = 10;
% 
% % Nw = rand(size(Nw));
% % Ns = rand(size(Ns));

% start of function

%% parameters
if nargin < 3
    Nshuffle = 0;
end
cutoff = 0.995;  % keep this much variance with PCA

%% test that inputs are normalized
n = size(Ns, 2);
if size(Nw, 2) ~= n
    error('Ns and Nw have different numbers of dimensions');
end

Ns_temp = normalize(zscore(Ns), 'norm');
Nw_temp = normalize(zscore(Nw), 'norm');

if sum((Ns_temp(:) - Ns(:)) .^2) > (0.01 .* Ns_temp.^2)
    warning('Ns was not normalized properly - normalizing now');
    Ns = Ns_temp;
end
if sum((Nw_temp(:) - Nw(:)) .^2) > (0.01 .* Nw_temp.^2)
    warning('Nw was not normalized properly - normalizing now');
    Nw = Nw_temp;
end

%% SVD on Ns and Nw
[~, Ss, Vs] = svd(Ns, 'econ');
[~, Sw, Vw] = svd(Nw, 'econ');

% discard PCs that cumulatively account for less than 1% of variance, i.e.
% rank-deficient dimensions
cumvar_s = cumsum(diag(Ss)) ./ sum(diag(Ss)); % cumulative variance
cumvar_w = cumsum(diag(Sw)) ./ sum(diag(Sw));

max_s = find(cumvar_s >= cutoff, 1, 'first');
max_w = find(cumvar_w >= cutoff, 1, 'first');

%% Zassenhaus algorithm to find shared basis for Ns and Nw
% https://en.wikipedia.org/wiki/Zassenhaus_algorithm
Vs_hat = Vs(:, 1:max_s);
Vw_hat = Vw(:, 1:max_w);

N_dim = size(Vs_hat, 1);
zassmat = [Vs_hat' Vs_hat'; Vw_hat' zeros(size(Vw_hat'))];
zassmat_rref = frref(zassmat);

basis_mat = [];
basis_idx = 1;
for idx = 1:size(zassmat_rref, 1)
    if all(zassmat_rref(idx, 1:N_dim) == 0) % this is a basis vector
        basis_mat(:, basis_idx) = zassmat_rref(idx, N_dim+1:end)';
        basis_idx = basis_idx + 1;
    end
end

J = orth(basis_mat); % orthonormal shared basis for Ns and Nw
k = size(J, 2);  % number of shared dimensions between Ns and Nw

disp(['Discarding ' num2str(n - k) ' low-variance dimensions']);


%% calculating spPCA

% maxizing X for X'*(Nw'*Nw-Ns'*Ns)*X / X'*(Ns'*Ns)*X

NwNw = Nw'*Nw;
NsNs = Ns'*Ns;

B = sqrtm(J' * NsNs * J);

JBinv = J / B;

[y, D] = eig(JBinv' * (NwNw - NsNs) * JBinv);

% sort according to decreasing eigenvalue
[D, sortidx] = sort(diag(D), 'descend');
y = y(:, sortidx);

X = normalize(JBinv * y, 'norm');

%% sorting eigenvectors according to descending variance

Stop = X' * (NwNw-NsNs) * X;
Sbot = X' * NsNs * X;

S.total = diag(Stop) ./ diag(Sbot);

%% shuffling to get SPV confidence intervals
%if Nshuffle > 0
%    error('shuffling does not work correctly');
%end

S.total_shuf = zeros(k, Nshuffle);
S.sleep      = zeros(k, 1);
S.sleep_shuf = zeros(k, Nshuffle);
S.awake      = zeros(k, 1);
S.awake_shuf = zeros(k, Nshuffle);

for shufidx = 1:Nshuffle  % if Nshuffle is zero, this won't run
    
%     Ns_shuf = Ns(:, randperm(size(Ns, 2))); % shuffling columns
%     Nw_shuf = Nw(:, randperm(size(Nw, 2))); % shuffling columns
    
    %THE SHUFFLING BELOW HAS TO BE DONE DIFFERENTLY, each cell at a time
    Ns_shuf = Ns(randperm(size(Ns, 1)), :); % shuffling rows
    Nw_shuf = Nw(randperm(size(Nw, 1)), :); % shuffling rows
    
    %[~, S_shuf] = spPCA_WminusSoverS(Ns_shuf, Nw_shuf);
    
    [~, S_shuf] = spPCA_WminusSoverS(Ns_shuf, Nw_shuf);
    
    S.total_shuf(:, shufidx) = S_shuf.total;
    S.sleep_shuf(:, shufidx) = S_shuf.sleep;
    S.awake_shuf(:, shufidx) = S_shuf.awake;
    disp(['Shuffle # ' num2str(shufidx) ' completed']);
end


%% gathering the variables to return

% info struct
info.info    = 'maximum sleep-preserved variance (SPV) dimensions';
info.n       = n;
info.n_info  = 'number of dimensions (e.g. units) in raw data';
info.k       = k;
info.k_info  = 'number of spPCs (shared dimensions accounting for 99% variance)';

% we already have X

% calculating the B's
clear B;
B.sleep = Ns * X;
B.awake = Nw * X;

% extracting the diagonal weight matrix
S.sleep = vecnorm(B.sleep)';
S.awake = vecnorm(B.awake)';

% normalizing the columns of the B's
B.sleep = B.sleep * diag(1./S.sleep);
B.awake = B.awake * diag(1./S.awake);
