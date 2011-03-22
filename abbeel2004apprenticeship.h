/* Compute an estimate of \mu using the monte carlo method,
   given a set of trajectories 
   \hat\mu = {1\over m }\sum_{i=0}^m
   \sum_{t=0}^\infty\gamma^t\psi(s_t^{(i)})
*/
gsl_matrix* monte_carlo_mu( gsl_matrix* D );

/* Abbeel and Ng's IRL algorithm (ANIRL), with the projection 
   method, monte-carlo estimation and LSPI as the MDP solver.
   Given a simulator and an expert's trace, returns the \omega 
   matrix which defines the optimal policy as found by LSPI
   under the reward R = \theta^T\psi. 
   Note that \omega \equiv \pi
*/
gsl_matrix* proj_mc_lspi_ANIRL( gsl_matrix* D_E,
				gsl_matrix* D,
				unsigned int m);
