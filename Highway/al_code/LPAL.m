%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function implements the LPAL algorithm from:
%
% Syed, U., Bowling, M., Schapire, R. E. (2008) "Apprenticeship Learning Using Linear Programming"
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
% X: The NA x 1 occupancy measure of the output policy, where NA is the number of state-action pairs. X(i) is the occupancy measure of state-action pair i.
%
% MA: The 1 x K "feature expectations" of the output policy. MA(i) is the expected cumulative discounted value for the ith feature when following the output policy (and when starting at the initial state distribution).
%
% TT: Running time of the linear program.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [X, MA, TT] = LPAL(THETA, F, GAMMA, E, INIT_FLAG)

[NA, N] = size(THETA);
[dummy, K] = size(F);
A = NA / N;
if (strcmp(INIT_FLAG, 'first'))
	init = zeros(N, 1);
	init(1) = 1;
elseif (strcmp(INIT_FLAG, 'uniform'))
	init = ones(N, 1) ./ N;
end

% For convenience, make features a (constant) function of action.
F_long = zeros(NA, K);
for i=1:K
    F_long(:, i) = reshape(kron(ones(1, A), F(:, i))', NA, 1);
end

% This matrix encodes the Bellman flow constraints
M = kron(eye(N), ones(1, A)) - GAMMA * THETA';

t1 = cputime;

cvx_begin
	variable X(NA);
	variable X_max;
	minimize X_max;
	subject to
		ones(K, 1) * X_max >= E' - F_long' * X;	
		init == M * X;
		X >= 0;
cvx_end

TT = cputime - t1;

MA = X' * F_long;
