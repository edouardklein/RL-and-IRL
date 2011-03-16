#include <gsl/gsl_matrix.h>
#include "GridWorld.h"
#include "utils.h"
#include "LSPI.h"
#include "greedy.h"
#include "GridWorld_simulator.h"
#include "abbeel2004apprenticeship.h"
#include "LSTDmu.h"
#define D_FILE_NAME "Samples.dat"
#define TRANS_WIDTH 7 
#define K (GRID_HEIGHT*GRID_WIDTH*4) /* dim(\phi) */
#define P (GRID_HEIGHT*GRID_WIDTH) /* dim(\psi) */
#define S 2
#define A 1
#define GAMMA 0.9
#define GAMMA_LSPI 0.9
#define EPSILON 0.1
#define EPSILON_LSPI 0.1

gsl_matrix* phi( gsl_matrix* sa ){
  gsl_matrix* answer = gsl_matrix_calloc( K, 1 );
  unsigned int x = (unsigned int)gsl_matrix_get( sa, 0, 0 );
  unsigned int y = (unsigned int)gsl_matrix_get( sa, 0, 1 );
  unsigned int a = (unsigned int)gsl_matrix_get( sa, 0, 2 );
  unsigned int index = (y-1)*GRID_WIDTH*4 + (x-1)*4 + a-1;
  gsl_matrix_set( answer, index, 0, 1.0 );
  return answer;
}

gsl_matrix* psi( gsl_matrix* s ){
  gsl_matrix* answer = gsl_matrix_calloc( P, 1 );
  unsigned int x = (unsigned int)gsl_matrix_get( s, 0, 0 );
  unsigned int y = (unsigned int)gsl_matrix_get( s, 0, 1 );
  unsigned int index = (y-1)*GRID_WIDTH + (x-1);
  gsl_matrix_set( answer, index, 0, 1.0 );
  return answer;
}

int main( void ){
  gsl_matrix* D = file2matrix( D_FILE_NAME, TRANS_WIDTH );
  gsl_matrix* omega_0 = gsl_matrix_calloc( K, 1 );
  /*As a side effect, this sets most of  the global variables 
    needed by greedy, and consequently by the simulator*/
  gsl_matrix* omega_expert = lspi( D, K, S, A, &phi, 
				   GAMMA, EPSILON, omega_0 );
  unsigned int M;
  for(M = 5; M<=35; M+=15 ){
    g_mOmega = gsl_matrix_alloc( K, 1 );
    gsl_matrix_memcpy( g_mOmega, omega_expert );
    g_mActions = file2matrix( ACTION_FILE, g_iA );
    g_iNb_samples = D->size1;
    gsl_matrix* D_expert = gridworld_simulator( M );
    gsl_matrix_free( g_mOmega );
    gsl_matrix_free( g_mActions );
    gsl_matrix* omega_imitation =
      proj_mc_lspi_ANIRL( D_expert, &gridworld_simulator, D, S,
  			  A, K, M, GAMMA, GAMMA_LSPI, EPSILON,
  			  EPSILON_LSPI, &phi, &psi );
    gsl_matrix_free( omega_imitation );
    gsl_matrix_free( D_expert );
  }
  for(M = 1; M<=100; M+=20 ){
    g_mOmega = gsl_matrix_alloc( K, 1 );
    gsl_matrix_memcpy( g_mOmega, omega_expert );
    g_mActions = file2matrix( ACTION_FILE, g_iA );
    g_iNb_samples = 0;
    gsl_matrix* D_expert = gridworld_simulator( M );
    gsl_matrix_free( g_mOmega );
    gsl_matrix_free( g_mActions );
    gsl_matrix* omega_lstd = 
      proj_lstd_lspi_ANIRL( D_expert, D_expert, S, A, K, P, GAMMA, 
			    GAMMA_LSPI, EPSILON, EPSILON_LSPI,
			    &phi, &psi );
    gsl_matrix_free( omega_lstd );
  }
  return 0;
}
