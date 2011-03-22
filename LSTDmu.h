/* Abbeel and Ng's IRL algorithm (ANIRL), with the projection 
   method, LSTDMu estimation and LSPI as the MDP solver.
   Given an expert's trace, returns the \omega 
   matrix which defines the optimal policy as found by LSPI
   under the reward R = \theta^T\psi. 
   Note that \omega \equiv \pi
*/ 
gsl_matrix* proj_lstd_lspi_ANIRL( gsl_matrix* D_E, 
				  gsl_matrix* D );
