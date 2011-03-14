/* Implements the LSPI algorithm from 
   \cite{lagoudakis2003least}.

   Matrix orientation is what is specified in ../Survey
*/

#define ACTION_FILE "actions.mat"
#define NB_ITERATIONS_MAX 20


gsl_matrix* lspi( gsl_matrix* D, unsigned int k, 
		  unsigned int s, unsigned int a,
		  gsl_matrix* (*phi)(gsl_matrix*),
		  double gamma, double epsilon,
		  gsl_matrix* omega_0 );
