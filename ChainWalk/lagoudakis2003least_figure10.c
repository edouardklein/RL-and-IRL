/*
  This program reproduces the figure 10 of 
  \cite{lagoudakis2003least}, but with variance.

  The input files are expected to have the following format :
  s a s' r eoe
*/
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <math.h>
#include "LSPI.h"
#include "utils.h"
#define D_WIDTH 5 /* s a s' r eoe, one number each */

#define S_CARD 4 /*State space cardinal*/
#define N_EXP  1000 /* Number of experiences on which we */
                    /* compute the mean and the var*/
#define D_PREFIX "Samples" /* Prefix for the files containing */
                           /* the transitions*/
#define ACTION_FILE "actions.mat"

gsl_matrix* phi( gsl_matrix* sa ){
  gsl_matrix* answer = gsl_matrix_calloc( 6, 1 );
  int index = (int)gsl_matrix_get(sa,0,1) * 3; //Action 0 for 
                                               //left, 1 for 
                                               //right
  gsl_matrix_set( answer, index, 0, 1. );
  gsl_matrix_set( answer, index+1, 0, gsl_matrix_get(sa,0,0) );
  gsl_matrix_set( answer, index+2, 0,
		 gsl_matrix_get(sa,0,0)*gsl_matrix_get(sa,0,0));
  return answer;
}

unsigned int g_iS = 1; /*State space dimension*/
unsigned int g_iA = 1; /*Action space dimension*/
unsigned int g_iK = 6; /*Feature space dimension*/
double g_dGamma_lstdq =  0.9; /*Discount factor*/
double g_dEpsilon_lspi = 0.1; /*Halt criterion*/
gsl_matrix* g_mOmega = NULL; //Omega
gsl_matrix* (*g_fPhi)(gsl_matrix*) = &phi; //\phi
gsl_matrix* g_mActions = NULL; //All actions, one per line
unsigned int g_iIt_max_lspi = 20;
double g_dLambda_lstdQ = 0.5; //Regularization influences
//variance in the final curve (a lot). Try 0, then 0.1, 
//then 0.5

int main( void ){
  gsl_matrix* omega_0 = gsl_matrix_calloc( g_iK, 1 );
  gsl_matrix* mQ = gsl_matrix_alloc( 1, 1 );
  gsl_matrix* values_L = gsl_matrix_calloc( N_EXP, S_CARD );
  gsl_matrix* values_R = gsl_matrix_calloc( N_EXP, S_CARD );
  g_mActions = file2matrix( ACTION_FILE, g_iA );

  for( unsigned int i = 0; i < N_EXP ; i++ ){
    //    fprintf( stderr, "LSPI on sample set %d\n", i );	
    char D_name[1024];
    sprintf( D_name, "%s%04d", D_PREFIX, i+1 );
    gsl_matrix* D = file2matrix( D_name, D_WIDTH );
    gsl_matrix* omega_star = lspi( D, omega_0 );
    for( unsigned int s = 1; s <= 4; s++ ){
      gsl_matrix* sa = gsl_matrix_alloc( 1, g_iS+g_iA );
      gsl_matrix_set( sa, 0, 0, (double)s );
      gsl_matrix_set( sa, 0, 1, 0. );//Left
      gsl_matrix* phi_sa = phi( sa );
      gsl_blas_dgemm( CblasTrans, CblasNoTrans, 
		      1.0, omega_star, phi_sa, 
		      0.0, mQ );
      gsl_matrix_set( values_L, i, s-1, 
		      gsl_matrix_get( mQ, 0, 0 ) );
      gsl_matrix_free( phi_sa );     
      gsl_matrix_set( sa, 0, 1, 1. );//Right
      phi_sa = phi( sa );
      gsl_blas_dgemm( CblasTrans, CblasNoTrans, 
		      1.0, omega_star, phi_sa, 
		      0.0, mQ );
      gsl_matrix_set( values_R, i, s-1, 
		      gsl_matrix_get( mQ, 0, 0 ) );
      
    }
    gsl_matrix_free( omega_star );
    gsl_matrix_free( D );
  }
  
  for( unsigned int s = 1; s <= 4; s++ ){
    double mean_L = 0;
    double mean_R = 0;
    for( unsigned int i = 0 ; i < N_EXP ; i++ ){
      mean_L += gsl_matrix_get( values_L, i, s-1 );
      mean_R += gsl_matrix_get( values_R, i, s-1 );
    }
    mean_L /= (double)N_EXP;
    mean_R /= (double)N_EXP;
    double var_L = 0;
    double var_R = 0;
    for( unsigned int i = 0 ; i < N_EXP ; i++ ){
      var_L += pow( gsl_matrix_get( values_L, i, s-1 ) - mean_L,
		    2 );
      var_R += pow( gsl_matrix_get( values_R, i, s-1 ) - mean_R,
		    2 );
    }
    var_L /= (double)N_EXP;
    var_R /= (double)N_EXP;

    printf("%d %lf %lf %lf %lf\n",
	   s, mean_L, var_L, mean_R, var_R );

  } 
  
  
  return 0;
}
