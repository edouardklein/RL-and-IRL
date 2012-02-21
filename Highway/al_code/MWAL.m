%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function implements the MWAL algorithm from:
%
% Syed, U., Schapire, R. E. (2007) "A Game-Theoretic Approach to Apprenticeship Learning"
%
% Here's a description of the parameters:
% 
% Input:
%
% THETA: The NA x N transition matrix, where NA is the number of state-action pairs, and A is the number of actions. THETA(i, j) is the probability of transitioning to state j under state-action pair i. NB: THETA should be sparse.
% 
% F: The N x K feature matrix, where N is the number of states, and K is the number of features. F(i, j) is the jth feature value for the ith state.
% 
% GAMMA: The discount factor, which must be a real number in [0, 1).
% 
% T: The number of iterations to run the algorithm. More iterations yields better results.
% 
% E: The 1 x K vector of "feature expectations" for the expert's policy. E(i) is the expected cumulative discounted value for the ith feature when following the expert's policy (with respect to initial state distribution).
%
% INIT_FLAG: If this is 'first', then initial state distribution is concentrated at state 1. If this is 'uniform', then initial state distribution is uniform
% 
% Output:
% 
% PP: The T x N matrix of output policies. PP(i, j) is the action taken in the jth state by the ith output policy. To achieve the guarantees of the algorithm, one must, at time 0, choose to follow exactly one of the output policies, each with probability 1/T. 
% 
% MM: The T x K matrix of output "feature expectations". MM(i, j) is the expected cumulative discounted value for the jth feature when following the ith output policy (and when starting at the initial state distribution).
%
% ITER: A 1 x T vector of value iteration iteration counts. ITER(i) is the number of iterations used by the ith invocation of value iteration.
% 
% TT: A 1 x T vector of value iteration running times. TT(i) is the number of seconds used by the ith invocation of value iteration.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [PP, MM, ITER, TT] = MWAL(THETA, F, GAMMA, T, E, INIT_FLAG)

[N, K] = size(F);
B = 1 ./ (1 + sqrt(2 * log(K) / T));

W = ones(K, 1);

% Choose initial features expectations randomly
VV = rand(N, K);
VV = sparse(VV);

for i=1:T
    disp(['MWAL iteration = ', num2str(i)]);
    disp(fix(clock));
    t1 = cputime;
    w = W ./ sum(W);
    [P, M, VV, ITER(i)] = opt_policy_and_feat_exp(THETA, F, GAMMA, w, INIT_FLAG, VV);
    X = (((1-GAMMA).*(M - E)) + 2.*ones(1, K))./4;
    W = W .* (B.^X');
    if (i == 1)
        TT(i) = cputime - t1;
    else
        TT(i) = TT(i-1) + cputime - t1;
    end
    PP(i, :) = P;
    MM(i, :) = M;
    disp(['w diff norm = ', num2str(norm(w - (W ./ sum(W))))]);
end
