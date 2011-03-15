/* Abbeel and Ng's IRL algorithm (ANIRL), with the projection 
   method, LSTDMu estimation and LSPI as the MDP solver.
   Given an expert's trace, returns the \omega 
   matrix which defines the optimal policy as found by LSPI
   under the reward R = \theta^T\psi. 
   Note that \omega \equiv \pi
*/
gsl_matrix* 
proj_lstd_lspi_ANIRL( gsl_matrix* expert_trans,
		      gsl_matrix* D,
		      unsigned int s, unsigned int a,
		      unsigned int k, unsigned int p,
		      double gamma, double gamma_lspi,
		      double epsilon, double epsilon_lspi,
		      gsl_matrix* (*phi)(gsl_matrix*),
		      gsl_matrix* (*psi)(gsl_matrix*));
