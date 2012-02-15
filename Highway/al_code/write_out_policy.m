%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function writes a policy to a text file, in a format suitable for my Perl scripts.
%
% Here's a description of the parameters:
% 
% Input:
% 
% P: The 1 x N vector that describes the policy. P(i) is the action taken in the ith state by the policy.
% 
% --- or ---
%
% P: The N x A matrix that describes the policy. P(i, j) is the probability of taking the jth action in the ith state.
% 
% Output (file):
%
% policy.dat: File containing N lines, with each line containing A comma-separated values, where N is the number of states and A is the number of actions. . The jth value in the ith line is the probability of taking the jth action in the ith state.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function write_out_policy(P)

if (isvector(P))
	A = max(P);

	for a=1:A
	   AA(a, find(P == a)) = 1;
	end

	csvwrite('policy.dat', AA');
else
	csvwrite('policy.dat', P)
end
