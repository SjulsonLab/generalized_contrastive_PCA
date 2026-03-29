function [B, S, X] = sparse_gcPCA(Ra, Rb, gcPCAversion, varargin)
% function [B, S, X] = sparse_gcPCA(Ra, Rb, gcPCAversion, varargin)
%
% This function performs sparse generalized contrastive PCA (gcPCA), which takes
% the input matrices Ra and Rb and finds the dimensions that differ most
% between them with penalty terms to create sparseness in the loadings.
% gcPCA is analogous to PCA/SVD, which finds dimensions that
% maximize variance in the data matrix Z, decomposing it into (U * S * V'),
% where U contains scores, V contains loadings, and S is a diagonal matrix
% showing how much variance each PC accounts for. Instead, gcPCA decomposes
% Ra and Rb into (Ba * Sa * X') and (Bb * Sb * X'), respectively. Unlike in
% PCA, Ba, Bb, and X are not generally orthogonal
%
% INPUTS
% Ra             -- (p1 x N) matrix of data (N features, p1 datapoints)
% Rb             -- (p2 x N) matrix of data (N features, p2 datapoints)
% gcPCAversion   -- version of gcPCA to use
% Nshuffle       -- (optional) number of times to shuffle for null
%                   distribution of S
% Ncalc          -- (optional) maximum number of gcPCs to calculate (useful
%                   only for orthogonal gcPCA, which is iterative)
% normalize      -- (optional) logical variable to normalize the data or
%                   not, default true. Remeber to at least center your data
%                   if turning this off.
%
% OUTPUTS
% B (struct)     -- gcPCA scores (different for Ra and Rb)
% S (struct)     -- amount of variance accounted for by each gcPC
% X              -- gcPCA loadings (shared between Ra and Rb)
%
% There are currently six versions of gcPCA, which find
% dimensions maximizing different objective functions:
%
% v2.0: maximizes Ra/Rb, X not constrained to be orthogonal
% v2.1: maximizes Ra/Rb, X constrained to be orthogonal
%
% v3.0: maximizes (Ra+Rb)/Rb, X not constrained to be orthogonal
% v3.1: maximizes (Ra+Rb)/Rb, X constrained to be orthogonal
%
% v4.0: maximizes (Ra+Rb)/(Ra+Rb), X not constrained to be orthogonal
% v4.1: maximizes (Ra+Rb)/(Ra+Rb), X constrained to be orthogonal
%
%
% Eliezyer de Oliveira, 2025-01-07



%% checking parameters and input arguments
pars = inputParser;
pars.addRequired('Ra');
pars.addRequired('Rb');
pars.addRequired('gcPCAversion');
pars.addOptional('Nshuffle', 0, @isnumeric);
pars.addOptional('normalize', true, @islogical);
pars.addOptional('Nsparse', 2, @isnumeric);
pars.addOptional('lasso_penalty', exp(linspace(log(1e-2),log(1),10)), @isnumeric);
pars.addOptional('ridge_penalty', 0, @isnumeric);
pars.addOptional('alpha', 1, @isnumeric);
pars.addOptional('maxcond', 10^13, @isnumeric); % you shouldn't need to change this, but it's condition number to regularize the denominator matrix if it's ill-conditioned
pars.addOptional('maxiter', 1000, @isnumeric);
pars.addOptional('tol', 1e-5, @isnumeric);

pars.parse('Ra', 'Rb', 'gcPCAversion', varargin{:});

maxcond = pars.Results.maxcond;
Nsparse = pars.Results.Nsparse;
alpha = pars.Results.alpha;
Nshuffle = pars.Results.Nshuffle;
normalize_flag = pars.Results.normalize;
lasso_penalty = pars.Results.lasso_penalty;
ridge_penalty = pars.Results.ridge_penalty;
alpha = pars.Results.alpha;
maxiter = pars.Results.maxiter;
tol = pars.Results.tol;

%% Step 1: normalize inputs if necessary
N = size(Rb, 2); % number of dimensions
if size(Ra, 2) ~= N
    error('Ra and Rb have different numbers of dimensions');
end

if normalize_flag
    Rb_temp = normalize(zscore(Rb), 'norm');
    Ra_temp = normalize(zscore(Ra), 'norm');
    
    if sum((Rb_temp(:) - Rb(:)) .^2) > (0.01 .* Rb_temp.^2)
        warning('Rb was not normalized properly - normalizing now');
        Rb = Rb_temp;
    end
    if sum((Ra_temp(:) - Ra(:)) .^2) > (0.01 .* Ra_temp.^2)
        warning('Ra was not normalized properly - normalizing now');
        Ra = Ra_temp;
    end
    clear Rb_temp Ra_temp
end

%% Step 2: do SVD and discard dimensions if necessary
p = min(size(Ra, 1), size(Rb, 1)); % we use the p from whichever dataset has fewer datapoints
N_gcPCs = min(p, N); % number of gcPCs to calculate. If p < N (meaning
% fewer datapoints than dimensions), we will only calculate p gcPCs

% doing SVD on the combined dataset. The goal here is to discard dimensions
% that have near-zero variance in *both* Ra and Rb
RaRb = [Ra; Rb];
tol = max(size(RaRb)) * eps(norm(RaRb)); % default cutoff used by rank()
[~, Sab, J] = svd(RaRb, 'econ'); % calculate SVD on combined data

Sab = diag(Sab);
if sum(Sab>tol) < N_gcPCs % data is rank-deficient
    warning('Input data is rank-deficient! Discarding dimensions; cannot shuffle.');
    N_gcPCs = sum(Sab>tol);
    Nshuffle = 0; % can't shuffle because it's not a valid null
end

% number of dimensions to return is either the amount requested by user or
% the amount of available dimensions, whichever is smaller
% N_gcPCs = min(N_gcPCs, Nsparse);

Jorig = J(:, 1:N_gcPCs); % using the first N_gcPCs dimensions
J = Jorig; % J shrinks every round, but Jorig is the first-round's J
clear RaRb Sab


%% Step 3: solving sparse gcPCA

if gcPCAversion == 1
    RaJ = Ra * J; % projecting into lower-D subspace spanned by J
    RbJ = Rb * J;
    
    JRaRaJ = RaJ'*RaJ / (size(RaJ,1)-1);
    JRbRbJ = RbJ'*RbJ / (size(RbJ,1)-1);
    obj_info = 'Ra - alpha*Rb';
    clear RaJ RbJ
    
    sigma = JRaRaJ - alpha*JRbRbJ;
    [y, D] = eig(sigma);
    D = diag(D);
    
    % Calculating only the number of dimensions requested by user
    n_gcpcs_pos = sum(D > 0);
    if (n_gcpcs_pos - Nsparse) >= 0
        n_gcpcs_pos = Nsparse;
        n_gcpcs_neg = 0;
    else
        temp = Nsparse - n_gcpcs_pos;
        n_gcpcs_neg = temp;
    end
    
    % Separating positive and negative eigenvalues
    Dpos = D;
    Dneg = D;
    
    %separating positive and negative eigenvalues
    Dpos(Dpos<0) = 0;
    Dneg(Dneg>0) = 0;
    
    % Square root matrix of the positive and 'negative' eigenvalues
    alpha_pos = max(Dpos)/maxcond;  % fixing the condition number
    theta_pos = y * diag(sqrt(Dpos + alpha_pos)) * y';
    
    alpha_neg = max(Dneg)/maxcond;  % fixing the condition number
    theta_neg = y * diag(sqrt(-Dneg + alpha_neg)) * y';
    
    % if there is any positive eigenvalue
    if n_gcpcs_pos>0
        final_pos_loadings = [];
        for lambda_idx = 1:length(lasso_penalty)
            lambda = lasso_penalty(lambda_idx);
            Bf = J_variable_projection(theta_pos, J, 'k', n_gcpcs_pos, 'alpha', lambda, 'beta', ridge_penalty, 'maxiter', maxiter, 'tol', tol);
            final_pos_loadings = [final_pos_loadings, normalize(Bf, 'norm')];
        end
        
    else
        final_pos_loadings = [];
    end
    
    % if there is any negative eigenvalue
    if n_gcpcs_neg>0
        final_neg_loadings = [];
        for lambda_idx = 1:length(lasso_penalty)
            lambda = lasso_penalty(lambda_idx);
            Bf = J_variable_projection(theta_neg, J, 'k', n_gcpcs_neg, 'alpha', lambda, 'beta', ridge_penalty, 'maxiter', maxiter, 'tol', tol);
            final_neg_loadings = [final_neg_loadings, normalize(Bf, 'norm')];
        end
        
    else
        final_neg_loadings = [];
    end
    
    % rearranging the PCs by vector and concatenating
    final_loadings = [];
    for lambda_idx = 1:length(lasso_penalty)
        if n_gcpcs_pos>0 && n_gcpcs_neg>0
            array_idx = Nsparse*(lambda_idx-1)+1:Nsparse*lambda_idx;
            final_loadings = [final_loadings , cat(2, final_pos_loadings(:,array_idx), final_neg_loadings(:,array_idx))];
        elseif n_gcpcs_pos == 0 && n_gcpcs_neg>0
            array_idx = Nsparse*(lambda_idx-1)+1:Nsparse*lambda_idx;
            final_loadings = [final_loadings, final_neg_loadings(:,array_idx)];
        else
            array_idx = Nsparse*(lambda_idx-1)+1:Nsparse*lambda_idx;
            final_loadings = [final_loadings, final_pos_loadings(:,array_idx)];
        end
        
    end
    
else
    
    % calculate numerator and denominator for objective function
    RaJ = Ra * J; % projecting into lower-D subspace spanned by J
    RbJ = Rb * J;
    
    JRaRaJ = RaJ'*RaJ / (size(RaJ,1)-1);
    JRbRbJ = RbJ'*RbJ / (size(RbJ,1)-1);
    if gcPCAversion == 2 % calculating gcPCA using Ra/Rb objective function
        numerator = JRaRaJ;
        denominator = JRbRbJ;
        obj_info = 'Ra ./ Rb';
        
    elseif gcPCAversion == 3 % calculating gcPCA using (Ra-Rb)/Rb objective function
        numerator = JRaRaJ - JRbRbJ;
        denominator = JRbRbJ;
        obj_info = '(Ra-Rb) ./ Rb';
        
    elseif gcPCAversion == 4 % calculating gcPCA using (Ra-Rb)/(Ra+Rb) objective function
        numerator = JRaRaJ - JRbRbJ;
        denominator = JRaRaJ + JRbRbJ;
        obj_info = '(Ra-Rb) ./ (Ra+Rb)';
    end
    
    %  Define numerator and denominator according to the method requested
    if gcPCAversion == 2
        numerator = JRaRaJ;
        denominator = JRbRbJ;
        obj_info = 'Ra / Rb';
    elseif gcPCAversion == 3
        numerator = JRaRaJ - JRbRbJ;
        denominator = JRbRbJ;
        obj_info = '(Ra-Rb) / Rb';
    elseif gcPCAversion == 4
        numerator = JRaRaJ - JRbRbJ;
        denominator = JRaRaJ + JRbRbJ;
        obj_info = '(Ra-Rb) / (Ra+Rb)';
    else
        error('Version input not recognized, please pick between v1-v4')
    end
    
    % getting the square root matrix of denominator
    [e, d] = eig(denominator);
    M = e * sqrt(d) * e';  % getting square root matrix M
    Minv = inv(M);
    
    sigma = Minv'*numerator*Minv;
    
    %
    % Getting square root matrix of sigma
    [v,w] = eig(sigma);
    
    
    % off setting the eigenvalues to be positive definite
    new_w = diag(diag(w) + 2);  % adding 2 to make it positive definite
    theta_pos = v * sqrt(new_w) * v';
    
    final_loadings = [];
    for lambda_idx = 1:length(lasso_penalty)
        lambda = lasso_penalty(lambda_idx);
        Bf = J_M_variable_projection(theta_pos, J, M, 'k', Nsparse, 'alpha', lambda, 'beta', ridge_penalty, 'maxiter', maxiter, 'tol', tol);
        final_loadings = [final_loadings, normalize(Bf, 'norm')];
    end
end

sparse_loadings=reshape(final_loadings,[size(final_loadings,1),Nsparse,length(lasso_penalty)]);
[~,n,p] = size(sparse_loadings); % getting loadings matrix dimensions

% calculating scores and values for condition A
temp_ra_scores = [];
temp_ra_values = [];
for sload_idx = 1:p
    sload = sparse_loadings(:,:,sload_idx);
    temp = Ra*sload;
    temp_norm = vecnorm(temp);
    temp_norm(temp_norm == 0) = 1;
    temp_ra_scores = [temp_ra_scores, temp./temp_norm];
    temp_ra_values = [temp_ra_values, vecnorm(temp)];
end

Ra_scores_ = reshape(temp_ra_scores,size(temp_ra_scores,1),n,p);
Ra_values_ = reshape(temp_ra_values,n,p);

% calculating scores and values for condition B
temp_rb_scores = [];
temp_rb_values = [];
for sload_idx = 1:p
    sload = sparse_loadings(:,:,sload_idx);
    temp = Rb*sload;
    temp_norm = vecnorm(temp);
    temp_norm(temp_norm == 0) = 1;
    temp_rb_scores = [temp_rb_scores, temp./temp_norm];
    temp_rb_values = [temp_rb_values, vecnorm(temp)];
end

Rb_scores_ = reshape(temp_rb_scores,[size(temp_rb_scores,1),n,p]);
Rb_values_ = reshape(temp_rb_values,n,p);

% organize output
B.gcPCAversion = gcPCAversion;
for idx = 1:p
    B.a{idx} = squeeze(Ra_scores_(:,:,idx));
end
B.a_info = 'Scores for Ra on gcPCA, each cell is a lasso penalty parameter';
for idx = 1:p
    B.b{idx} = squeeze(Rb_scores_(:,:,idx));
end
B.b_info = 'Scores for Rb on gcPCA, each cell is a lasso penalty parameter';

S.gcPCAversion = gcPCAversion;
S.objective = obj_info;
for idx = 1:p
    S.a{idx} = Ra_values_(:,idx);
end
S.a_info = 'gcPCA values for Ra, each cell is a lasso penalty parameter';
for idx = 1:p
    S.b{idx} = Rb_values_(:,idx);
end
S.b_info = 'gcPCA values for Rb, each cell is a lasso penalty parameter';
S.lasso_penalty = lasso_penalty;
S.ridge_penalty = ridge_penalty;

for idx = 1:p
    X{idx} = squeeze(sparse_loadings(:,:,idx));
end

end

