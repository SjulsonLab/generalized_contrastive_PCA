function [B, S, X] = gcPCA(Ra, Rb, gcPCAversion, varargin)

% function [B, S, X] = gcPCA(Ra, Rb, gcPCAversion, varargin)
%
% This function performs generalized contrastive PCA (gcPCA), which takes
% the input matrices Ra and Rb and finds the dimensions that differ most
% between them. gcPCA is analogous to PCA/SVD, which finds dimensions that
% maximize variance in the data matrix Z, decomposing it into (U * S * V'),
% where U contains scores, V contains loadings, and S is a diagonal matrix
% showing how much variance each PC accounts for. Instead, gcPCA decomposes
% Ra and Rb into (Ba * Sa * X') and (Bb * Sb * X'), respectively. Unlike in
% PCA, Ba, Bb, and X are not generally orthogonal, but we can constrain X
% to be orthogonal (versions 2.1/3.1/4.1).
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
% Luke Sjulson, 2023-06-28


%% checking parameters and input arguments
pars = inputParser;
pars.addRequired('Ra');
pars.addRequired('Rb');
pars.addRequired('gcPCAversion');
pars.addOptional('Nshuffle', 0, @isnumeric);
pars.addOptional('normalize', true, @islogical);
pars.addOptional('Ncalc', Inf, @isnumeric);
pars.addOptional('alpha', 1, @isnumeric);
pars.addOptional('maxcond', 10^13, @isnumeric); % you shouldn't need to change this, but it's condition number to regularize the denominator matrix if it's ill-conditioned
pars.addOptional('rPCAkeep',1,@isnumeric); %this parameter is to control the amount of variance you want the original PCs to explain, 1 for full data
pars.parse('Ra', 'Rb', 'gcPCAversion', varargin{:});
maxcond = pars.Results.maxcond;
Ncalc = pars.Results.Ncalc;
alpha = pars.Results.alpha;
Nshuffle = pars.Results.Nshuffle;
normalize_flag = pars.Results.normalize;
rPCAkeep = pars.Results.rPCAkeep; %this is important if you want to exclude very low variance and unstable PCs from your data

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

%% keep only the PCs that explain r amount of variance
if rPCAkeep < 1
   [ua,sa,va] = svd(Ra,'econ');
   [ub,sb,vb] = svd(Rb,'econ');
   
   na = cumsum(diag(sa)./sum(sa(:)))<rPCAkeep; % n pcs to keep
   nb = cumsum(diag(sb)./sum(sb(:)))<rPCAkeep; % n pcs to keep
   Ra = ua(:,na)*sa(na,na)*va(:,na)';
   Rb = ub(:,nb)*sb(nb,nb)*vb(:,nb)';
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
if gcPCAversion == 2 || gcPCAversion == 3 || gcPCAversion == 4 && ~isinf(Ncalc) % if we're not iterating and Ncalc has been set
    warning('Ncalc is only relevant if using orthogonal gcPCA. Will calculate full set of gcPCs.');
elseif gcPCAversion == 2.1 || gcPCAversion == 3.1 || gcPCAversion == 4.1 % we're iterating and may need to limit how many N_gcPCs to calculate
    N_gcPCs = min(N_gcPCs, Ncalc);
end

Jorig = J(:, 1:N_gcPCs); % using the first N_gcPCs dimensions
J = Jorig; % J shrinks every round, but Jorig is the first-round's J
clear RaRb Sab

%% Step 3: doing gcPCA

% if we want to have orthogonal gcPCs, we need to iterate multiple times
if gcPCAversion == 2.1 || gcPCAversion == 3.1 || gcPCAversion == 4.1
    Niter = N_gcPCs;
else
    Niter = 1;
end

if gcPCAversion == 1
    RaRa = Ra'*Ra / (size(Ra,1)-1);
    RbRb = Rb'*Rb / (size(Rb,1)-1);
    
    JRaRaJ = J'*RaRa*J; % projecting into lower-D subspace spanned by J
    JRbRbJ = J'*RbRb*J;
    obj_info = 'Ra - alpha*Rb';
    clear RaJ RbJ
    
    sigma = JRaRaJ - alpha*JRbRbJ;
    [y, D] = eig(sigma);
    [~, sortidx] = sort(diag(D), 'descend');
    clear D
    y = y(:, sortidx);
    
    % the matrix of gcPCs
    X = normalize(J * y, 'norm');
    
else
    track_index = zeros(1,Niter);
    count_first_index = 1;
    RaRa = Ra'*Ra / (size(Ra,1)-1);
    RbRb = Rb'*Rb / (size(Rb,1)-1);
    for idx = 1:Niter % if we are not calculating orthogonal gcPCs, it only iterates once
        
        % calculate numerator and denominator for objective function
        JRaRaJ = J'*RaRa*J; % projecting into lower-D subspace spanned by J
        JRbRbJ = J'*RbRb*J;
        
        if gcPCAversion == 2 || gcPCAversion == 2.1 % calculating gcPCA using Ra/Rb objective function
            numerator = JRaRaJ; % calculating the lower-D covariance matrices
            denominator = JRbRbJ;
            obj_info = 'Ra ./ Rb';
            
        elseif gcPCAversion == 3 || gcPCAversion == 3.1 % calculating gcPCA using (Ra-Rb)/Rb objective function
            numerator = JRaRaJ - JRbRbJ;
            denominator = JRbRbJ;
            obj_info = '(Ra-Rb) ./ Rb';
            
        elseif gcPCAversion == 4 || gcPCAversion == 4.1 % calculating gcPCA using (Ra-Rb)/(Ra+Rb) objective function
            numerator = JRaRaJ - JRbRbJ;
            denominator = JRaRaJ + JRbRbJ;
            obj_info = '(Ra-Rb) ./ (Ra+Rb)';
        end
        
        % calculating the gcPCs
        M = sqrtm(denominator);
        clear denominator
        [y, D] = eig(M \ numerator / M);
        
        clear numerator
        [D, sortidx] = sort(diag(D), 'descend');
        clear D
        y = y(:, sortidx);
        
        % the matrix of gcPCs
        Xtemp = normalize(J / M * y, 'norm');
        clear M
        
        % copy results to X
        if idx == 1
            X = Xtemp;
            Xorth = []; % orthogonal version of X
        end
%         Xorth(:, idx) = Xtemp(:, 1);
        if mod(idx,2) == 1
            Xorth(:, idx) = Xtemp(:, 1);
            track_index(idx) = count_first_index;
            count_first_index = count_first_index+1;
        elseif mod(idx,2) == 0
            Xorth(:, idx) = Xtemp(:, end);
            track_index(idx) = size(Xtemp,2)+count_first_index-1;
        end
        clear Xtemp
        
        % shrinking J (find an orthonormal basis for the subspace of J orthogonal
        % to the X vectors we have already collected)
        [J, ~] = svd(Jorig - Xorth * (Xorth' * Jorig), 'econ');
        J = J(:,1:Niter-idx);
        
    end
    
    if Niter > 1 % returning the orthogonalized version
        [~,I] = sort(track_index);
        X = Xorth(:,I);
    end
    clear Xorth
end
%% gathering variables to return

% calculating the B's here so we can clear Ra and Rb from RAM prior to
% shuffling
B.gcPCAversion = gcPCAversion;
B.b = Rb * X;
B.a = Ra * X;

% extracting the diagonal weight matrices (I shouldn't call these Ra and Rb)
Ra = vecnorm(B.a)';
Rb = vecnorm(B.b)';

% normalizing the columns of the B's
B.a = B.a * diag(1./Ra);
B.b = B.b * diag(1./Rb);

% initializing S
S.gcPCAversion = gcPCAversion;

%% shuffling to get gcPCA confidence intervals
if Nshuffle > 0

    % initializing struct so fields will be in correct order
    S.objective = obj_info;
    S.objval = [];
    S.objval_info = 'value of the objective function';
    S.objval_shuf = zeros(N_gcPCs, Nshuffle);
    S.objval_shuf_info = 'value of objective function on shuffled data';

    S.a = [];
    S.a_info = 'diagonal entries of S_a';
    S.a_shuf = [];
    S.a_shuf_info = 'diagonal entries of S_a on shuffled data';
    S.b_shuf = [];
    S.b_shuf_info = 'diagonal entries of S_b on shuffled data';

    S.b = [];
    S.b_info = 'diagonal entries of S_b';

    S.a      = zeros(N_gcPCs, 1);
    S.a_shuf = zeros(N_gcPCs, Nshuffle);
    S.b      = zeros(N_gcPCs, 1);
    S.b_shuf = zeros(N_gcPCs, Nshuffle);

    for a = 1:size(Ra,2)
        Ra_shuf(:,a) = Ra(randperm(size(Ra, 1)), a); % shuffling rows
        Rb_shuf(:,a) = Rb(randperm(size(Rb, 1)), a); % shuffling rows
    end
    clear Ra Rb % we will keep only the shuffled version in RAM
    
%     Rb_shuf = Rb(:, randperm(size(Rb, 2))); % shuffling columns
%     clear Rb
    for shufidx = 1:Nshuffle
        
    for a = 1:size(Ra_shuf,2)
        Ra_shuf(:,a) = Ra_shuf(randperm(size(Ra_shuf, 1)), a); % shuffling rows
        Rb_shuf(:,a) = Rb_shuf(randperm(size(Rb_shuf, 1)), a); % shuffling rows
    end
    
        [~, S_shuf] = gcPCA(Ra_shuf, Rb_shuf, gcPCAversion, 'maxcond', maxcond, 'Ncalc', Ncalc);

%         Rb_shuf = Rb_shuf(:, randperm(size(Rb_shuf, 2))); % shuffling columns
%         [~, S_shuf] = gcPCA(Ra, Rb_shuf, gcPCAversion, 'maxcond', maxcond, 'Ncalc', Ncalc);

        S.objval_shuf(:, shufidx) = S_shuf.objval;
        S.a_shuf(:, shufidx) = S_shuf.a;
        S.b_shuf(:, shufidx) = S_shuf.b;
        disp(['Shuffle # ' num2str(shufidx) ' completed']);        
        
    end 

end

%% finish calculating S

S.objective = obj_info;
eval(['S.objval = ' obj_info ';']); % kludgy 
S.objval_info = 'value of the objective function';
S.a = Ra;
S.a_info = 'diagonal entries of S_a';
S.b = Rb;
S.b_info = 'diagonal entries of S_b';



