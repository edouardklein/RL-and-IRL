#include <gsl/gsl_matrix.h>
/* #include <gsl/gsl_blas.h> */
#include "LSTDQ.h"
#include "utils.h"
#include "greedy.h"
/* #include "LSPI.h" */
/* #include "criteria.h" */
#include "RL_Globals.h"

gsl_matrix* lspi( gsl_matrix* D, gsl_matrix* omega_0 ){
  unsigned int nb_iterations = 0;
  gsl_matrix* omega = gsl_matrix_alloc( g_iK, 1 );
  g_mOmega = omega;
  //\omega'\leftarrow \omega_0
  gsl_matrix* omega_dash = gsl_matrix_alloc( g_iK, 1 );
  gsl_matrix_memcpy( omega_dash, omega_0 );
  double norm;
  //Repeat
  do{
    //\omega \leftarrow \omega'
    gsl_matrix_memcpy( omega, omega_dash );
    //\omega' \leftarrow lstd_q(D,k,\phi,\gamma,\omega)
    gsl_matrix_free( omega_dash );
    omega_dash = lstd_q( D, &greedy_policy );
    //until ( ||\omega'-\omega|| < \epsilon )
    norm = diff_norm( omega_dash, omega );
    nb_iterations++;
  }while( norm >= g_dEpsilon_lspi && 
	  nb_iterations < g_iIt_max_lspi);
  gsl_matrix_free( omega );
  //We return omega' and not omega
  return omega_dash;
}
