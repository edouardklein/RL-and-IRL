
#include <gsl/gsl_matrix.h>
#include "LSTDQ.h"
#include "utils.h"
#include "greedy.h"
#include "LSPI.h"
#include "RL_Globals.h"

gsl_matrix* lspi( gsl_matrix* D, gsl_matrix* omega_0 ){
  unsigned int nb_iterations = 0;
  gsl_matrix* omega = gsl_matrix_alloc( g_iK, 1 );
  g_mOmega = omega;

  gsl_matrix* omega_dash = gsl_matrix_alloc( g_iK, 1 );
  gsl_matrix_memcpy( omega_dash, omega_0 );
  double norm;

  do{

     gsl_matrix_memcpy( omega, omega_dash );

     gsl_matrix_free( omega_dash );
     omega_dash = lstd_q( D, &greedy_policy );

     norm = diff_norm( omega_dash, omega );
     nb_iterations++;
   }while( norm >= g_dEpsilon_lspi && 
	   nb_iterations < g_iIt_max_lspi);
   gsl_matrix_free( omega );

  return omega_dash;
}
