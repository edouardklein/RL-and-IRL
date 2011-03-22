/* Implements the LSTDQ algorithm from 
   \cite{lagoudakis2003least}.

   Matrix orientation is what is specified in ../Survey
*/
gsl_matrix* lstd_q( gsl_matrix* D, 
		    gsl_matrix* (*pi)(gsl_matrix*) );
