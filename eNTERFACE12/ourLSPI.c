#define _POSIX_C_SOURCE 1
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <math.h>
#include "utils.h"
#include "LSPI.h"
#include "greedy.h"
#include "RL_Globals.h"
#define D_FILE_NAME "RandomSamples.mat"
#define TRANS_WIDTH 15
#define ACTION_FILE "actions.mat"

//FIXME : those are not useful here, but it won't compile without
double g_dGamma_lafem = 0;
unsigned int g_iNb_episodes = -1;

unsigned int g_iS = 6;
unsigned int g_iA = 1;
unsigned int g_iIt_max_lspi = 50;

unsigned int g_iACTION_CARD = 4;
unsigned int g_aCards[6] = {3,4,3,3,3,3};
    
unsigned int g_iP = (972); /* dim(\psi) */
unsigned int g_iK = (972*4); /* dim(\phi) */

unsigned int prod( unsigned int* tab, unsigned int start, unsigned int end ){
  double answer = 1.;
  for( unsigned int i = start; i<end; i++ ){
    answer *= tab[i];
  }
  return answer;
}

gsl_matrix* psi( gsl_matrix* s ){
  gsl_matrix* answer = gsl_matrix_calloc( g_iP, 1 );
  unsigned int index = 0;
  for(unsigned int i=0;i < g_iS; i++){
    index += gsl_matrix_get(s,0,i) * prod(g_aCards,0,i );
  }
  gsl_matrix_set(answer,index,0, 1.);
  return answer;
}


gsl_matrix* phi( gsl_matrix* sa ){
  gsl_matrix* answer = gsl_matrix_calloc( g_iK, 1 );
  gsl_matrix_view vs = gsl_matrix_submatrix( sa, 0,0, 1, g_iS );
  unsigned int action = (unsigned int)gsl_matrix_get( sa, 0, g_iS );
  gsl_matrix* mpsi = psi( &vs.matrix );
  unsigned int index = action*g_iP;
  for( unsigned int i = 0; i < g_iP ; i++){
    gsl_matrix_set( answer, index + i, 0, 
		    gsl_matrix_get( mpsi, i, 0 ) );
  }
  return answer;
}

gsl_matrix* (*g_fPhi)(gsl_matrix*) = &phi;
gsl_matrix* g_mOmega = NULL;
double g_dLambda_lstdQ = 0.1;
double g_dGamma_lstdq =  0.9;
double g_dEpsilon_lspi = 0.01;
gsl_matrix* g_mActions = NULL; 

unsigned int g_iMax_episode_len = -1;

int main (int argc, char *argv[]){
  if (argc != 2){
    printf("usage : %s <name of file with theta inside>\n (%d arguments given) ",argv[0],argc);
    exit( 1 );
  }
  char* theta_file = argv[1];

  gsl_matrix* theta_lafem = file2matrix( theta_file, 1 );
  gsl_matrix* D = file2matrix( D_FILE_NAME, TRANS_WIDTH );
  gsl_matrix* new_reward = gsl_matrix_alloc( 1, 1 );
  for( int i=0; i<D->size1; i++ ){
    gsl_matrix_view vstate = gsl_matrix_submatrix( D, i, 0, 1, 2 );
    gsl_matrix* mPsi = psi( &(vstate.matrix) );
    gsl_blas_dgemm( CblasTrans, CblasNoTrans, 1.0, theta_lafem, mPsi, 0., new_reward );
    gsl_matrix_set( D, i, 5, gsl_matrix_get( new_reward, 0,0 ) );
    gsl_matrix_free( mPsi );
  }

  g_mActions = file2matrix( ACTION_FILE, g_iA );
  gsl_matrix* omega_0 = gsl_matrix_calloc( g_iK, 1 );
  gsl_matrix* omega_lafem = lspi( D, omega_0 );
  for( int i=0; i<omega_lafem->size1; i++ ){
    printf("%f \n",gsl_matrix_get(omega_lafem,i,0));
  }  
  return 0;
}
