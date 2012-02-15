%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function computes the optimal policy with respect to a particular reward function, and simultaneously computes the "feature expectations" for that policy.
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
% w: A K x 1 vector that is a convex combination, i.e. the components of w are nonnegative and sum to 1.
%
% INIT_FLAG: Specifies initial state distribution. Either 'uniform' or 'first' (i.e. concentrated at first state)
%
% VV: Initial per-state feature expectations. Intended to be carried over from previous invocations. 
% 
% Output:
% 
% P: The 1 x N vector that describes the optimal policy with respect to the reward function R = F*w. P(i) is the action taken in the ith state by the optimal policy.
% 
% M: The 1 x K vector of "feature expectations" for the optimal policy. M(i) is the expected cumulative discounted value for the ith feature when following the optimal policy (and when starting at state 1).
%
% VV: Final per-state feature expectations. Intended to carry over to future invocations. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [P, M, VV, ITER] = opt_policy_and_feat_exp(THETA, F, GAMMA, w, INIT_FLAG, VV)

%tol = max([0.001 * abs(sum(w)), 0.001]);
tol = 0.001;
disp(['tol = ', num2str(tol)]);
[NA, N] = size(THETA);
A = NA/N;
if (strcmp(INIT_FLAG, 'first'))
	init = zeros(1, N);
	init(1) = 1;
elseif (strcmp(INIT_FLAG, 'uniform'))
	init = ones(1, N) ./ N;
end
[dummy, K] = size(F);
F_long = zeros(N*A, K);

for i=1:K
    F_long(:, i) = reshape(kron(ones(1, A), F(:, i))', N*A, 1);
end

% Choose initial features expectations randomly, if necessary
if (isempty(VV))
	VV = rand(N, K);
	VV = sparse(VV);
end

V = (VV * w)';

% Conserve memory
clear F;
F_long = sparse(F_long);
w = sparse(w);

delta = tol + 1;

ITER = 0;

while (delta > tol)

    Q = F_long + GAMMA * THETA * VV;
    [V_new, P] = max(reshape((Q * w), A, N));
    AA = zeros(A, N, K);
    for a=1:A
        AA(a, find(P == a), :) = 1;
    end
    Q = reshape(full(Q), A, N, K);
    VV = squeeze(sum(AA .* Q));

    % Conserve memory
    VV = sparse(VV);

    delta = max(abs(V - V_new));
    disp(['tol = ', num2str(tol), '; ValIter delta = ' num2str(delta)]);

    V = full(V_new);

    ITER = ITER + 1;
end

M = init * full(VV);
