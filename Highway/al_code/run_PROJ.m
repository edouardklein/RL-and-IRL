function [PP, MM, ITER, TT, c] = run_PROJ

% Make the feature value matrix and the transition matrix
F = make_F;
THETA = make_THETA;

% Setup the other parameters
GAMMA = 0.9;
T = 500;
E = [5.25, 4.15, 5];
%E = [7.5, 5, 5];
%E = [9.5, -0.8967, 0];
K = length(E);

% Run the projection algorithm
[PP, MM, ITER, TT] = PROJ(THETA, F, GAMMA, T, E, 'first');

% Solve a quadratic program to determine the mixing coefficients (requires CVX)
cvx_begin
	variable u(K);
	variable c(T);
	minimize ( norm(E' - u) );
	subject to
		MM' * c == u;
		ones(1, T) * c  == 1;
		c >= 0;
cvx_end

% Choose a policy at random according to the mixing coefficients
C(1) = c(1);
for i=2:T
	C(i) = C(i-1) + c(i);
end
r = rand;
i = find(r <= C, 1);

% Write out that policy
write_out_policy(PP(i, :));
