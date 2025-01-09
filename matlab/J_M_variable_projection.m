function [loadings] = J_M_variable_projection(theta, J, M, varargin)
    % functioon [loadings] = J_M_variable_projection(theta, J, M, varargin)
    %
    % This function performs variable projection on the data in theta. It
    % returns the loadings of the first principal component of the data with sparseness
    % constrains given by parameters 'alpha' and 'beta', that corresnpond to the
    % lasso and ridge penalty parameters in elastic net regression.
    %
    % INPUTS:
    % theta: data matrix of size n x p, where n is the number of samples and p is the number of variables
    % J: the principal components used to reduce the data complexity in gcPCA
    % M: the square root matrix used in the gcPCA solution
    %
    % OPTIONAL INPUTS:
    % k: number of components to extract (default: inf for all components)
    % alpha: lasso penalty parameter (default: 1e-4)
    % beta: ridge penalty parameter (default: 1e-4)
    % maxiter: maximum number of iterations (default: 1000)
    % tol: tolerance for convergence (default: 1e-5)
    % verbose: print progress to screen (default: True)
    
    
    
    %% checking parameters and input arguments
    pars = inputParser;
    pars.addRequired('theta');
    pars.addRequired('J');
    pars.addRequired('M');
    pars.addOptional('k', inf, @isnumeric);
    pars.addOptional('alpha', 1e-4, @isnumeric);
    pars.addOptional('beta', 1e-4, @isnumeric);
    pars.addOptional('maxiter', 1000, @isnumeric);
    pars.addOptional('tol', 1e-5, @isnumeric);
    pars.addOptional('verbose',true, @isboolean);
    pars.parse(theta, J, M, varargin{:});
    
    k = pars.Results.k;
    alpha = pars.Results.alpha;
    beta = pars.Results.beta;
    maxiter = pars.Results.maxiter;
    tol = pars.Results.tol;
    verbose = pars.Results.verbose;
    
    [~,S,V] = svd(theta, 'econ');
%     S = diag(S);
    Dmax = S(1,1);
    B = V(:,1:k);
    
    VD = V * S;
    VD2 = V * (S.^2);
    
    % Set tuning parameters
    alpha = alpha * Dmax^2;
    beta = beta * Dmax^2;
    
    nu = 1 / (Dmax^2 + beta);
    kappa = nu * alpha;
    
    old_obj = 0;
    improvement = inf;
    
    % Apply Variable Projection Solver
    VD2_Vt = VD2 * V';
    JMinv = J * inv(M);
    MJt = M * J';
    for iter = 1:maxiter
        % Update A: X'XB = UDV'
        Z = VD2_Vt * B;
        [U_Z,~,V_Z] = svd(Z, 'econ');
        A = U_Z * V_Z';
    
        grad = (VD2 * (V' * (A-B)) - beta*B);  % Gradient step in the original space
        B_temp = JMinv*B + nu * JMinv*grad;  % Passing it to the feature space
        B_temp_f = B_temp;
        
        % l1 soft_thresholding
        Bf = zeros(size(B_temp_f));
        Bf(B_temp_f > kappa) = B_temp_f(B_temp_f > kappa) - kappa;
        Bf(B_temp_f <= -kappa) = B_temp_f(B_temp_f <= -kappa) + kappa;

        B = MJt*Bf;  % Passing it back to the original space
    
        R = VD' - (VD'*B*A');  % Residuals
        obj_value = 0.5 * sum(R(:).^2) + alpha * sum(abs(B(:))) + 0.5 * beta * sum(B(:).^2);
        obj = obj_value;
    
        % Check convergence, break if objective is not improving
        if iter > 1
            improvement = abs(obj - old_obj);
        end
    
        if improvement < tol
            break
        end
        old_obj = obj;
        
        if verbose && mod(iter, 10) == 0
            fprintf('Iteration %d: Objective = %f, Improvement = %f\n', iter, obj, improvement);
        end
    end
    
    loadings = Bf;
end
