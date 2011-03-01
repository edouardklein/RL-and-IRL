/* Implements the LSPI algorithm from 
   \cite{lagoudakis2003least}.

   Matrix orientation is what is specified in ../Survey
*/


gsl_matrix* lspi( gsl_matrix* D, unsigned int k, 
		  unsigned int s, unsigned int a,
		  gsl_matrix* (*phi)(gsl_matrix*),
		  double gamma, double epsilon,
		  gsl_matrix* omega_0 );
