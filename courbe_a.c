#include <gsl/gsl_matrix.h>
#include "GridWorld.h"
#include "utils.h"
#include "LSPI.h"
#include "greedy.h"
#include "GridWorld_simulator.h"
#include "abbeel2004apprenticeship.h"
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
#define M_QUALITY 1000 /* Number of episodes when */
                /* computing the quality of a policy*/
#define M_EXP 100 /* Number of episodes when */
       /* creating D_expert */
#define M 100 /* Number of episodes for MC */

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
  g_mOmega = gsl_matrix_alloc( K, 1 );
  gsl_matrix_memcpy( g_mOmega, omega_expert );
  g_mActions = file2matrix( ACTION_FILE, g_iA );
  gsl_matrix* D_expert = gridworld_simulator( M_EXP );
  gsl_matrix_free( g_mOmega );
  gsl_matrix_free( g_mActions );
  double q = quality( gridworld_simulator, S, A, 
		      omega_expert, M_QUALITY );
  fprintf(stderr, "Quality of the expert : %lf\n", q );
  gsl_matrix* omega_imitation = 
    proj_mc_lspi_ANIRL( D_expert, &gridworld_simulator, D, S, A,
			K, M, GAMMA, GAMMA_LSPI, EPSILON, 
			EPSILON_LSPI, &phi, &psi );
  q = quality( gridworld_simulator, S, A, 
	       omega_imitation, M_QUALITY );
  fprintf(stderr, "Quality of the imitation : %lf\n", q );
  return 0;
}
