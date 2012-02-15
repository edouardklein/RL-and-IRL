function [MA, TT] = run_LPAL

% Make the feature value matrix and the transition matrix 
THETA = make_THETA;
F = make_F;

% Setup the other parameters
GAMMA = 0.9;
E = [5.25, 4.15, 5];
%E = [7.5, 5, 5];
%E = [0, 0, 0];
[NA, N] = size(THETA);
A = NA / N;

% Run the LPAL algorithm
[X, MA, TT] = LPAL(THETA, F, GAMMA, E, 'first');

% Construct stationary policy from occupancy measure
P = zeros(N, A);
for s=1:N
	sum = 0;
	for a=1:A
		sa = A * (s - 1) + a;
		sum = sum + X(sa);
	end

	for a=1:A
		sa = A * (s - 1) + a;
		P(s, a) = X(sa) / sum;
	end
end

% Write out that policy
write_out_policy(P);
