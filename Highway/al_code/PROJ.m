%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function implements the projection algorithm from:
%
% Abbeel, P., Ng, A. (2004) "Apprenticeship Learning via Inverse Reinforcement Learning"
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
% PP: The T x N matrix of output policies. PP(i, j) is the action taken in the jth state by the ith output policy. To achieve the guarantees of the algorithm, one must, at time 0, choose to follow the ith output policy with probability c(i), where c(i) is determined by solving a quadratic program.
% 
% MM: The T x K matrix of output "feature expectations". MM(i, j) is the expected cumulative discounted value for the jth feature when following the ith output policy (and when starting at the initial state distribution).
%
% ITER: A 1 x T vector of value iteration iteration counts. ITER(i) is the number of iterations used by the ith invocation of value iteration.
% 
% TT: A 1 x T vector of value iteration running times. TT(i) is the number of seconds used by the ith invocation of value iteration.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [PP, MM, ITER, TT] = PROJ(THETA, F, GAMMA, T, E, INIT_FLAG)

[N, K] = size(F);

w(1, :) = rand(1, K);
w(1, :) = w(1, :) ./ norm(w(1, :));

% Choose initial features expectations randomly
VV = rand(N, K);
VV = sparse(VV);

for i=2:(T+1)
    disp(['PROJ iteration = ', num2str(i)]);
    disp(fix(clock));
    t1 = cputime;
    
    [PP(i-1, :), MM(i-1, :), VV, ITER(i-1)] = opt_policy_and_feat_exp(THETA, F, GAMMA, w(i-1, :)', INIT_FLAG, VV);
    if (i == 2)
        BM(1, :) = MM(1, :);
        w(2, :) = E - MM(1, :);
    else
        x = MM(i-1, :) - BM(i-2, :);
        y = E - BM(i-2, :);
        BM(i-1, :) = BM(i-2, :) + ((x*y') ./ (x*x')).*x;
        w(i, :) = E - BM(i-1, :);
    end
    if (i == 2)
        TT(i-1) = cputime - t1;
    else
        TT(i-1) = TT(i-2) + cputime - t1;
    end
    disp(['PROJ Norm = ', num2str(norm(w(i, :)))]);
    disp(['w diff norm = ', num2str(norm(w(i, :) - w(i-1, :)))]);
end

