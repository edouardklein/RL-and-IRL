function [PP, MM, ITER, TT, c] = run_MWAL

% Make the feature value matrix and the transition matrix 
F = make_F;
THETA = make_THETA;

% Setup the other parameters
GAMMA = 0.9;
T = 500;
E = [5.25, 4.15, 5];
%E = [7.5, 5, 5];
%E = [9.5, -0.8967, 0];

% Run the MWAL algorithm
[PP, MM, ITER, TT] = MWAL(THETA, F, GAMMA, T, E, 'first');

% Determine the mixing coefficients (trivial)
c = ones(T, 1) ./ T;

% Choose a policy at random according to the mixing coefficients
C(1) = c(1);
for i=2:T
	C(i) = C(i-1) + c(i);
end
r = rand;
i = find(r <= C, 1);

% Write out that policy
write_out_policy(PP(i, :));
