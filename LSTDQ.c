
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include "RL_Globals.h"

gsl_matrix* lstd_q( gsl_matrix* D, 
		    gsl_matrix* (*pi)(gsl_matrix*) ){

  gsl_matrix* A = gsl_matrix_calloc( g_iK, g_iK );

  gsl_matrix* b = gsl_matrix_calloc( g_iK, 1 );

  for( unsigned int i=0; i < D->size1 ; i++ ){

     gsl_matrix_view sa = 
       gsl_matrix_submatrix( D, i, 0, 1, g_iS+g_iA );
     gsl_matrix* phi_sa = g_fPhi( &sa.matrix );
     gsl_matrix* sa_dash = gsl_matrix_alloc( 1, g_iS+g_iA );
     gsl_matrix_view sdash_dst = 
       gsl_matrix_submatrix( sa_dash, 0, 0, 1, g_iS );
     gsl_matrix_view sdash_src = 
       gsl_matrix_submatrix( D, i, g_iS+g_iA, 1, g_iS );
     gsl_matrix_memcpy( &sdash_dst.matrix, &sdash_src.matrix );
     gsl_matrix_view adash_dst = 
       gsl_matrix_submatrix( sa_dash, 0, g_iS, 1, g_iA );
     gsl_matrix* adash_src = pi( &sdash_src.matrix );
     gsl_matrix_memcpy( &adash_dst.matrix, adash_src );
     gsl_matrix* phi_dash = g_fPhi( sa_dash );
     gsl_matrix_scale( phi_dash, g_dGamma_lstdq );
     double eoe = gsl_matrix_get( D, i, g_iS+g_iA+g_iS+1 );
     gsl_matrix_scale( phi_dash, eoe );
     gsl_matrix* delta_phi = gsl_matrix_calloc( g_iK, 1 );
     gsl_matrix_memcpy( delta_phi, phi_sa );
     gsl_matrix_sub( delta_phi, phi_dash );
     gsl_matrix* deltaA = gsl_matrix_calloc( g_iK, g_iK );
     gsl_blas_dgemm( CblasNoTrans, CblasTrans, 1., 
		       phi_sa, delta_phi, 0., deltaA);
     gsl_matrix_add( A, deltaA );

     double r = gsl_matrix_get( D, i, g_iS+g_iA+g_iS );
     gsl_matrix_scale( phi_sa, r ); //phi_sa is now delta_b
     gsl_matrix_add( b, phi_sa );
     gsl_matrix_free( deltaA );
     gsl_matrix_free( delta_phi );
     gsl_matrix_free( phi_dash );
     gsl_matrix_free( adash_src );
     gsl_matrix_free( sa_dash );
     gsl_matrix_free( phi_sa );
   }

  gsl_matrix* lambdaI = gsl_matrix_alloc( A->size1, A->size2 );
  gsl_matrix_set_identity( lambdaI );
  gsl_matrix_scale( lambdaI, g_dLambda_lstdQ );
  gsl_matrix_add( A, lambdaI );
  gsl_matrix_free( lambdaI );
  gsl_vector_view b_v = gsl_matrix_column( b, 0 );
  gsl_matrix* omega_pi = gsl_matrix_alloc( g_iK, 1 );
  gsl_vector_view o_v = gsl_matrix_column( omega_pi, 0 );
  gsl_permutation* p = gsl_permutation_alloc( g_iK );
  int signum;
  gsl_linalg_LU_decomp( A, p, &signum );
  gsl_linalg_LU_solve( A, p, &b_v.vector, &o_v.vector );
  gsl_matrix_free( A );
  gsl_matrix_free( b );
  gsl_permutation_free( p );

  return omega_pi;
}
