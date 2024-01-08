function [B, S, X] = gcPCA(Za, Zb, gcPCAversion, varargin)

% function [B, S, X] = gcPCA(Za, Zb, gcPCAversion, varargin)
%
% This function performs generalized contrastive PCA (gcPCA), which takes
% the input matrices Za and Zb and finds the dimensions that differ most
% between them. gcPCA is analogous to PCA/SVD, which finds dimensions that
% maximize variance in the data matrix Z, decomposing it into (U * S * V'),
% where U contains scores, V contains loadings, and S is a diagonal matrix
% showing how much variance each PC accounts for. Instead, gcPCA decomposes
% Za and Zb into (Ba * Sa * X') and (Bb * Sb * X'), respectively. Unlike in
% PCA, Ba, Bb, and X are not generally orthogonal, but we can constrain X
% to be orthogonal (versions 2.1/3.1/4.1).
% 
% INPUTS
% Za             -- (p1 x N) matrix of data (N features, p1 datapoints)
% Zb             -- (p2 x N) matrix of data (N features, p2 datapoints)
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
% B (struct)     -- gcPCA scores (different for Za and Zb)
% S (struct)     -- amount of variance accounted for by each gcPC
% X              -- gcPCA loadings (shared between Za and Zb)
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
pars.addRequired('Za');
pars.addRequired('Zb');
pars.addRequired('gcPCAversion');
pars.addOptional('Nshuffle', 0, @isnumeric);
pars.addOptional('normalize', true, @islogical);
pars.addOptional('Ncalc', Inf, @isnumeric);
pars.addOptional('alpha', 1, @isnumeric);
pars.addOptional('maxcond', 10^13, @isnumeric); % you shouldn't need to change this, but it's condition number to regularize the denominator matrix if it's ill-conditioned
pars.addOptional('rPCAkeep',1,@isnumeric); %this parameter is to control the amount of variance you want the original PCs to explain, 1 for full data
pars.parse('Za', 'Zb', 'gcPCAversion', varargin{:});
maxcond = pars.Results.maxcond;
Ncalc = pars.Results.Ncalc;
alpha = pars.Results.alpha;
Nshuffle = pars.Results.Nshuffle;
normalize_flag = pars.Results.normalize;
rPCAkeep = pars.Results.rPCAkeep; %this is important if you want to exclude very low variance and unstable PCs from your data

%% Step 1: normalize inputs if necessary
N = size(Zb, 2); % number of dimensions
if size(Za, 2) ~= N
    error('Za and Zb have different numbers of dimensions');
end

if normalize_flag
    Zb_temp = normalize(zscore(Zb), 'norm');
    Za_temp = normalize(zscore(Za), 'norm');
    
    if sum((Zb_temp(:) - Zb(:)) .^2) > (0.01 .* Zb_temp.^2)
        warning('Zb was not normalized properly - normalizing now');
        Zb = Zb_temp;
    end
    if sum((Za_temp(:) - Za(:)) .^2) > (0.01 .* Za_temp.^2)
        warning('Za was not normalized properly - normalizing now');
        Za = Za_temp;
    end
    clear Zb_temp Za_temp
end

%% keep only the PCs that explain r amount of variance
if rPCAkeep < 1
   [ua,sa,va] = svd(Za,'econ');
   [ub,sb,vb] = svd(Zb,'econ');
   
   na = cumsum(diag(sa)./sum(sa(:)))<rPCAkeep; % n pcs to keep
   nb = cumsum(diag(sb)./sum(sb(:)))<rPCAkeep; % n pcs to keep
   Za = ua(:,na)*sa(na,na)*va(:,na)';
   Zb = ub(:,nb)*sb(nb,nb)*vb(:,nb)';
end
%% Step 2: do SVD and discard dimensions if necessary
p = min(size(Za, 1), size(Zb, 1)); % we use the p from whichever dataset has fewer datapoints
N_gcPCs = min(p, N); % number of gcPCs to calculate. If p < N (meaning
% fewer datapoints than dimensions), we will only calculate p gcPCs

% doing SVD on the combined dataset. The goal here is to discard dimensions
% that have near-zero variance in *both* Za and Zb
ZaZb = [Za; Zb];
tol = max(size(ZaZb)) * eps(norm(ZaZb)); % default cutoff used by rank()
[~, Sab, J] = svd(ZaZb, 'econ'); % calculate SVD on combined data

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
clear ZaZb Sab

%% Step 3: doing gcPCA

% if we want to have orthogonal gcPCs, we need to iterate multiple times
if gcPCAversion == 2.1 || gcPCAversion == 3.1 || gcPCAversion == 4.1
    Niter = N_gcPCs;
else
    Niter = 1;
end

if gcPCAversion == 1
    ZaJ = Za * J; % projecting into lower-D subspace spanned by J
    ZbJ = Zb * J;
    
    AA = ZaJ'*ZaJ;
    BB = ZbJ'*ZbJ;
    obj_info = 'Ra - alpha*Rb';
    clear ZaJ ZbJ
    
    sigma = AA - alpha*BB;
    [y, D] = eig(sigma);
    [~, sortidx] = sort(diag(D), 'descend');
    clear D
    y = y(:, sortidx);
    
    % the matrix of gcPCs
    X = normalize(J * y, 'norm');
    
else
    denom_well_conditioned = false;
    
    for idx = 1:Niter % if we are not calculating orthogonal gcPCs, it only iterates once
        
        % calculate numerator and denominator for objective function
        ZaJ = Za * J; % projecting into lower-D subspace spanned by J
        ZbJ = Zb * J;
        
        if gcPCAversion == 2 || gcPCAversion == 2.1 % calculating gcPCA using Ra/Rb objective function
            numerator = ZaJ'*ZaJ; % calculating the lower-D covariance matrices
            denominator = ZbJ'*ZbJ;
            obj_info = 'Ra ./ Rb';
            
        elseif gcPCAversion == 3 || gcPCAversion == 3.1 % calculating gcPCA using (Ra-Rb)/Rb objective function
            numerator = ZaJ'*ZaJ - ZbJ'*ZbJ;
            denominator = ZbJ'*ZbJ;
            obj_info = '(Ra-Rb) ./ Rb';
            
        elseif gcPCAversion == 4 || gcPCAversion == 4.1 % calculating gcPCA using (Ra-Rb)/(Ra+Rb) objective function
            numerator = ZaJ'*ZaJ - ZbJ'*ZbJ;
            denominator = ZaJ'*ZaJ + ZbJ'*ZbJ;
            obj_info = '(Ra-Rb) ./ (Ra+Rb)';
        end
        clear ZaJ ZbJ
        
        % diagonal loading the denominator matrix if it's ill-conditioned. If
        % we're iterating, we only need to test it until it's well-conditioned
        % once, and it will always be well-conditioned after that
        if denom_well_conditioned == false
            denom_SVspectrum = svd(denominator);
            if max(denom_SVspectrum)/min(denom_SVspectrum) > maxcond % matrix ill-conditioned
                warning('Denominator matrix ill-conditioned. Regularizing...')
                alpha = max(denom_SVspectrum)/maxcond - min(denom_SVspectrum); % approximately correct, close enough
                denominator = denominator + alpha * eye(size(denominator));
            else
                denom_well_conditioned = true;
            end
            clear denom_SVspectrum alpha
        end
        
        % calculating the gcPCs
        M = sqrtm(denominator);
        clear denominator
        [y, D] = eig(M \ numerator / M);
        %     [y, D] = eig(inv(M) * numerator * inv(M));
        clear numerator
        %     y = real(y); % there can be tiny imaginary parts due to numerical instability
        %     D = real(diag(D));
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
        Xorth(:, idx) = Xtemp(:, 1);
        clear Xtemp
        
        % shrinking J (find an orthonormal basis for the subspace of J orthogonal
        % to the X vectors we have already collected)
        [J, ~] = svd(Jorig - Xorth * (Xorth' * Jorig), 'econ');
        J = J(:,1:Niter-idx);
        
    end
    
    if Niter > 1 % returning the orthogonalized version
        % Xorig = X; % this is the original non-orthogonal X for comparison
        X = Xorth;
    end
    clear Xorth
end
%% gathering variables to return

% calculating the B's here so we can clear Za and Zb from RAM prior to
% shuffling
B.gcPCAversion = gcPCAversion;
B.b = Zb * X;
B.a = Za * X;

% extracting the diagonal weight matrices (I shouldn't call these Ra and Rb)
Ra = vecnorm(B.a)'; %luke's implementation
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

    for a = 1:size(Za,2)
        Za_shuf(:,a) = Za(randperm(size(Za, 1)), a); % shuffling rows
        Zb_shuf(:,a) = Zb(randperm(size(Zb, 1)), a); % shuffling rows
    end
    clear Za Zb % we will keep only the shuffled version in RAM
    
%     Zb_shuf = Zb(:, randperm(size(Zb, 2))); % shuffling columns
%     clear Zb
    for shufidx = 1:Nshuffle
        
    for a = 1:size(Za_shuf,2)
        Za_shuf(:,a) = Za_shuf(randperm(size(Za_shuf, 1)), a); % shuffling rows
        Zb_shuf(:,a) = Zb_shuf(randperm(size(Zb_shuf, 1)), a); % shuffling rows
    end
    
        [~, S_shuf] = gcPCA(Za_shuf, Zb_shuf, gcPCAversion, 'maxcond', maxcond, 'Ncalc', Ncalc);

%         Zb_shuf = Zb_shuf(:, randperm(size(Zb_shuf, 2))); % shuffling columns
%         [~, S_shuf] = gcPCA(Za, Zb_shuf, gcPCAversion, 'maxcond', maxcond, 'Ncalc', Ncalc);

        S.objval_shuf(:, shufidx) = S_shuf.objval;
        S.a_shuf(:, shufidx) = S_shuf.a;
        S.b_shuf(:, shufidx) = S_shuf.b;
        disp(['Shuffle # ' num2str(shufidx) ' completed']);        
        
    end 

end

%% finish calculating S

S.objective = obj_info;
eval(['S.objval = ' obj_info ';']); % kludgy af
S.objval_info = 'value of the objective function';
S.a = Ra;
S.a_info = 'diagonal entries of S_a';
S.b = Rb;
S.b_info = 'diagonal entries of S_b';



