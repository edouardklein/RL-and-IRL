#include <time.h>
#include <unistd.h>
#include <gsl/gsl_matrix.h> 
#include <math.h>
#include "utils.h"
#include "simulator.h"
#include "greedy.h"
#include "InvertedPendulum.h"
#define TRANS_WIDTH 7 /* s,a,s',r,e = 7*/

unsigned int g_iNb_samples = 0;

/* Simulate nbEpisodes episodes using greedy_policy()*/
gsl_matrix* inverted_pendulum_simulator( int nbEpisodes ){
  srand(time(NULL)+getpid()); rand(); rand();rand();
  gsl_matrix* transitions = gsl_matrix_alloc( nbEpisodes*g_iMax_episode_len, TRANS_WIDTH );
  // set the value of the pendulum in a state near the 
  // equilibrium
  double state_p; //position
  double state_v; //vitesse
  iv_init( &state_p, &state_v );
  gsl_matrix* state = gsl_matrix_alloc( 1, 2 );
  unsigned int j = 0; //Index in transitions
  for( unsigned int i = 0 ; i < nbEpisodes ; i++ ){
    unsigned int nb_steps = 0;
    int eoe = 1;
    while( eoe==1 ){
      nb_steps++;
      g_iNb_samples++;
      gsl_matrix_set( state, 0, 0, (double)state_p );
      gsl_matrix_set( state, 0, 1, (double)state_v );
      double next_state_p;
      double next_state_v;
      gsl_matrix* mAction = greedy_policy( state );
      unsigned int action = 
	(unsigned int)gsl_matrix_get( mAction, 0, 0 );
      gsl_matrix_free( mAction );
      double reward;
      iv_step( state_p, state_v, action, 
	       &next_state_p, &next_state_v, &reward, &eoe );
      if( nb_steps == g_iMax_episode_len ){
	eoe = 0;
      }
      gsl_matrix_set( transitions, j, 0, (double)state_p );
      gsl_matrix_set( transitions, j, 1, (double)state_v );
      gsl_matrix_set( transitions, j, 2, (double)action );
      gsl_matrix_set( transitions, j, 3, (double)next_state_p );
      gsl_matrix_set( transitions, j, 4, (double)next_state_v );
      gsl_matrix_set( transitions, j, 5, (double)reward );
      gsl_matrix_set( transitions, j, 6, (double)eoe );
      j++;
      if( eoe == 1 ){
	state_p = next_state_p;
	state_v = next_state_v;
      }else{
	iv_init( &state_p, &state_v );
      }
    }
  }
  gsl_matrix_free( state );
  gsl_matrix* answer = gsl_matrix_alloc( j, TRANS_WIDTH );
  gsl_matrix_view trans_v = 
    gsl_matrix_submatrix( transitions, 0, 0, j, TRANS_WIDTH );
  gsl_matrix_memcpy( answer, &trans_v.matrix );
  gsl_matrix_free( transitions );
  return answer;
}
