/* Implements the LSTDQ algorithm from 
   \cite{lagoudakis2003least}.

   Matrix orientation is what is specified in ../Survey
*/
gsl_matrix* lstd_q( gsl_matrix* D, unsigned int k,
		    unsigned int s, unsigned int a,
		    gsl_matrix* (*phi)(gsl_matrix*),
		    double gamma,
		    gsl_matrix* (*pi)(gsl_matrix*) );

