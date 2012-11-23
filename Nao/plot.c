
#define _POSIX_C_SOURCE 1
#include <gsl/gsl_matrix.h>
#include <math.h>
#include "InvertedPendulum.h"
#include "simulator.h"
#include "utils.h"
#include "LSPI.h"
#include "abbeel2004apprenticeship.h"
#include "LSTDmu.h"
#include "criteria.h"
#include "greedy.h"
#include "phipsi.h"

#include "RL_Globals.h"
#include "IRL_Globals.h"
//#define D_FILE_NAME "Samples.dat"
//#define TRANS_WIDTH 7
//#define ACTION_FILE "actions.mat"

#define ACTION_FILE "../resources/a.txt"
#define D_FILE_NAME "../resources/sas.txt"



/*gsl_matrix* initial_state( void ){
  gsl_matrix* answer = gsl_matrix_alloc( 1, 2 );
  double pos;
  double speed;
  for( unsigned int i=0; i<1; i++ ){
    iv_init( &pos, &speed );
    gsl_matrix_set( answer, i, 0, pos );
    gsl_matrix_set( answer, i, 1, speed );
  }
  return answer;
}*/

gsl_matrix* initial_state( void ){
  gsl_matrix* answer = file2matrix("../resources/s0.txt",3);

  return answer;
}


unsigned int g_iS = 3;
unsigned int g_iA = 3;
unsigned int g_iIt_max_lspi = 50;
gsl_matrix* (*g_fPhi)(gsl_matrix*) = &phi;
gsl_matrix* g_mOmega = NULL;
double g_dLambda_lstdQ = 0.1;
double g_dGamma_lstdq =  0.9;
double g_dEpsilon_lspi = 0.01;
double g_dLambda_lstdmu = 0.1;
double g_dGamma_anirl = 0.9;
double g_dEpsilon_anirl = 0.01;
unsigned int g_iIt_max_anirl = 2;
//gsl_matrix* g_mActions = NULL;
gsl_matrix* (*g_fPsi)(gsl_matrix*) = &psi;
gsl_matrix* (*g_fSimulator)(int) = &inverted_pendulum_simulator;
gsl_matrix* (*g_fS_0)(void) = &initial_state;
unsigned int g_iMax_episode_len = 20; //Shorter episodes are
//better for MC, as s_0 is seen more often

/*
int plot_main( void ){
  fprintf(stderr,"Training the expert...");
  fflush( NULL );
  gsl_matrix* D = file2matrix( D_FILE_NAME, TRANS_WIDTH );
  g_mActions = file2matrix( ACTION_FILE, g_iA );
  gsl_matrix* omega_0 = gsl_matrix_calloc( g_iK, 1 );
  gsl_matrix* omega_expert = lspi( D, omega_0 );
  g_mOmega_E = omega_expert;
  expert_just_set();
  fprintf(stderr,"done\n");

  gsl_matrix* D_expert;
  int D_len[] = {1,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,3000};
  //Cette courbe illustre l'influence de
  // la taille de D sur les deux variantes
  for( int j=0; j<21;j++){
    g_iNb_samples = 0;
    g_mOmega =  omega_expert;
    g_iMax_episode_len = 100;
    D_expert = inverted_pendulum_simulator( 1 );
    unsigned int nb_samples_exp = g_iNb_samples;
    gsl_matrix_view sub_D =
      gsl_matrix_submatrix( D, 0, 0, D_len[j], TRANS_WIDTH );
    g_iMax_episode_len = 100;
    gsl_matrix* omega_lstd =
      proj_lstd_lspi_ANIRL( D_expert, &sub_D.matrix );
    g_mOmega = omega_lstd;
    g_iNb_samples = 0;
    g_iMax_episode_len = 3000;
    gsl_matrix* discard = inverted_pendulum_simulator( 100 );
    gsl_matrix_free( discard );
    unsigned int mean_control_steps = g_iNb_samples/100;
    gsl_matrix_free( omega_lstd );
    printf("LSTD %d %d %d\n", nb_samples_exp,
	   D_len[j],  mean_control_steps );
    g_iMax_episode_len = 100;
    gsl_matrix* omega_mc =
      proj_mc_lspi_ANIRL( D_expert, &sub_D.matrix, 2000 );
    g_mOmega = omega_mc;
    g_iNb_samples = 0;
    g_iMax_episode_len = 3000;
    discard = inverted_pendulum_simulator( 100 );
    gsl_matrix_free( discard );
    mean_control_steps = g_iNb_samples/100;
    gsl_matrix_free( omega_mc );
    printf("MC %d %d %d\n", nb_samples_exp,
	   D_len[j],  mean_control_steps );

    gsl_matrix_free( D_expert );
    }

  gsl_matrix_free( g_mActions );
  gsl_matrix_free( omega_expert );
  expert_free();
  gsl_matrix_free( D );
  gsl_matrix_free( omega_0 );
  return 0;
}
*/
