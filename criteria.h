/* Different criteria for assessing the quality of 
   the imitation algorithms. */

/* "True" difference between the feature expectation
   of the expert and the one for the current policy :
   ||\mu_E-\mu||_2
   Computed via a big monte-carlo.
   expert_just_set() must be called before this function.
*/
double true_diff_norm( gsl_matrix* omega );

/* Must call this one after the global variable
   g_mOmega_E is set, so that its feature expectation
   and V(s_0) can be computed
*/
void expert_just_set();

/* Compute V(s0) for the given set of trajectories */
double value_func( gsl_matrix* D );


/* |V^E(s_0)-V^\pi(s_0)| for the current policy.
   Objective measurement for task transfer.
   Computed via a big monte_carlo.
*/
double true_V_diff( gsl_matrix* omega );
