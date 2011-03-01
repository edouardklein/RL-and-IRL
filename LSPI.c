#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include "LSTDQ.h"
#include "utils.h"
#define ACTION_FILE "actions.mat"
#define NB_ITERATIONS_MAX 20

unsigned int g_iS; //Dimension of S
unsigned int g_iA; //Dimension on A
unsigned int g_iK; //Number of features
gsl_matrix* g_mOmega = NULL; //Omega
gsl_matrix* (*g_fPhi)(gsl_matrix*); //\phi
gsl_matrix* g_mActions = NULL; //All actions, one per line


gsl_matrix* greedy_policy( gsl_matrix* state ){
  gsl_matrix* sa = gsl_matrix_alloc( 1, g_iS + g_iA );
  gsl_matrix_view s_dst = gsl_matrix_submatrix( sa, 0, 0, 
						1, g_iS );
  gsl_matrix_memcpy( &s_dst.matrix, state );
  gsl_matrix_view a_dst = gsl_matrix_submatrix( sa, 0, g_iS,
						1, g_iA );
  gsl_matrix_view a_src = 
    gsl_matrix_submatrix( g_mActions, 0, 0, 1, g_iA );
  gsl_matrix* best_action = gsl_matrix_alloc( 1, g_iA );
  gsl_matrix_memcpy( &a_dst.matrix, &a_src.matrix );
  gsl_matrix_memcpy( best_action, &a_src.matrix );
  gsl_matrix* phi_sa = g_fPhi( sa );
  gsl_matrix* mQ = gsl_matrix_alloc( 1, 1 );
  gsl_blas_dgemm( CblasTrans, CblasNoTrans, 
		  1.0, g_mOmega, phi_sa, 
		  0.0, mQ );
  gsl_matrix_free( phi_sa );
  double Q_max = gsl_matrix_get( mQ, 0, 0 );
  for( unsigned int i = 0 ; i < g_mActions->size1 ; i++ ){
    a_src = gsl_matrix_submatrix( g_mActions, i, 0, 1, g_iA );
    gsl_matrix_memcpy( &a_dst.matrix, &a_src.matrix );
    gsl_matrix* phi_sa = g_fPhi( sa );
    gsl_blas_dgemm( CblasTrans, CblasNoTrans, 
		    1.0, g_mOmega, phi_sa, 
		    0.0, mQ );
    double Q_current = gsl_matrix_get( mQ, 0, 0 );
    if( Q_current > Q_max ){
      Q_max = Q_current;
      gsl_matrix_memcpy( best_action, &a_src.matrix );
    }
    gsl_matrix_free( phi_sa );
  }
  gsl_matrix_free( mQ );
  return best_action;
}

gsl_matrix* lspi( gsl_matrix* D, unsigned int k, 
		  unsigned int s, unsigned int a,
		  gsl_matrix* (*phi)(gsl_matrix*),
		  double gamma, double epsilon,
		  gsl_matrix* omega_0 ){
  g_iA = a;
  g_iS = s;
  g_iK = k;
  g_fPhi = phi;
  g_mActions = file2matrix( ACTION_FILE, g_iA );
  g_mOmega = gsl_matrix_alloc( omega_0->size1, 
			       omega_0->size2 );
  unsigned int nb_iterations = 0;
  //\omega'\leftarrow \omega_0
  gsl_matrix* omega_dash = gsl_matrix_alloc( omega_0->size1,
					     omega_0->size2 );
  gsl_matrix_memcpy( omega_dash, omega_0 );
  double norm;
  //Repeat
  do{
    //\omega \leftarrow \omega'
    gsl_matrix_memcpy( g_mOmega, omega_dash );
    //\omega' \leftarrow lstd_q(D,k,\phi,\gamma,\omega)
    gsl_matrix_free( omega_dash );
    omega_dash = lstd_q( D, k, s, a, phi, gamma, 
			 &greedy_policy );
    //until ( ||\omega'-\omega|| < \epsilon )
    gsl_vector_view o_v = gsl_matrix_column( g_mOmega, 0 );
    gsl_vector_view o_d_v = gsl_matrix_column( omega_dash, 0 );
    gsl_vector* diff = gsl_vector_calloc( k );
    gsl_vector_memcpy( diff, &o_v.vector );
    gsl_vector_sub( diff, &o_d_v.vector );
    norm = gsl_blas_dnrm2( diff );
    nb_iterations++;
  }while( norm >= epsilon && nb_iterations < NB_ITERATIONS_MAX);
  gsl_matrix_free( g_mOmega );
  gsl_matrix_free( g_mActions );
  //We return omega' and not omega
  return omega_dash;
}
