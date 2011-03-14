/* Compute the mean reward over nb_ep episodes */
double quality( gsl_matrix* (*simulator)(int), 
		unsigned int s, unsigned int a, 
		gsl_matrix* omega, unsigned int nb_ep );
/* Abbeel and Ng's IRL algorithm (ANIRL), with the projection 
   method, monte-carlo estimation and LSPI as the MDP solver.
   Given a simulator and an expert's trace, returns the \omega 
   matrix which defines the optimal policy as found by LSPI
   under the reward R = \theta^T\psi. 
   Note that \omega \equiv \pi
*/
gsl_matrix* proj_mc_lspi_ANIRL( gsl_matrix* expert_trans,
				gsl_matrix* (*simulator)(int),
				gsl_matrix* D,
				unsigned int s, unsigned int a,
				unsigned int k, unsigned int m,
				double gamma, 
				double gamma_lspi,
				double epsilon,
				double epsilon_lspi,
				gsl_matrix* (*phi)(gsl_matrix*),
			       gsl_matrix* (*psi)(gsl_matrix*));
