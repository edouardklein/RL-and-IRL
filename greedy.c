
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include "RL_Globals.h"

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
  gsl_matrix_free( sa );
  return best_action;
}

double greedy_value_function( gsl_matrix* state ){
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
  gsl_matrix_free( best_action );
  return Q_max;
}
