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
#include "RL_Globals.h"
#include "IRL_Globals.h"
#include "greedy.h"
#define D_FILE_NAME "Samples.dat"
#define TRANS_WIDTH 7
#define ACTION_FILE "actions.mat"

unsigned int g_iK = (30); /* dim(\phi) */
unsigned int g_iP = (10); /* dim(\psi) */

gsl_matrix* phi( gsl_matrix* sa ){
  gsl_matrix* answer = gsl_matrix_calloc( g_iK, 1 );
  double position = gsl_matrix_get( sa, 0, 0 );
  double speed = gsl_matrix_get( sa, 0, 1 );
  unsigned int action = (unsigned int)gsl_matrix_get( sa, 0, 2 );
  unsigned int index = action*10;
  gsl_matrix_set( answer, index, 0, 1.0 );
  index++;
  for( int i = -1. ; i <= 1 ; i++ ){
    for( int j = -1 ; j <= 1 ; j++ ){
      double d_i = (double)i * PI/4.;
      double d_j = (double)j;
      gsl_matrix_set( answer, index, 0, 
		      exp(-(pow(position-d_i,2) + 
			    pow(speed-d_j,2))/2.));
      index++;
    }
  }
  return answer;
}

gsl_matrix* psi( gsl_matrix* s ){
  gsl_matrix* answer = gsl_matrix_calloc( g_iP, 1 );
  double position = gsl_matrix_get( s, 0, 0 );
  double speed = gsl_matrix_get( s, 0, 1 );
  unsigned int index = 0;
  gsl_matrix_set( answer, index, 0, 1.0 );
  index++;
  for( int i = -1. ; i <= 1 ; i++ ){
    for( int j = -1 ; j <= 1 ; j++ ){
      double d_i = (double)i * PI/4.;
      double d_j = (double)j;
      gsl_matrix_set( answer, index, 0,
		      exp(-(pow(position-d_i,2) + 
			    pow(speed-d_j,2))/2.));
      index++;
    }
  }
  return answer;
}

unsigned int g_iS = 2;
unsigned int g_iA = 1;
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
gsl_matrix* g_mActions = NULL; 
gsl_matrix* (*g_fPsi)(gsl_matrix*) = &psi;
gsl_matrix* (*g_fSimulator)(int) = &inverted_pendulum_simulator;
unsigned int g_iMax_episode_len = 30; //Shorter episodes are 
//better for MC, as s_0 is seen more often

int main( void ){
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
  gsl_matrix* omega_imitation;
  //La première courbe montre que sans matrice D, 
  //quelque soit la taille de la matrice D_E,
  //la variante LSTD ne fonctionne pas tandis que
  //la variante MC fonctionne
  int m_exp[] = {1,10,100};
  for( int i=0; i<3 ; i++ ){
    g_iNb_samples = 0;
    g_mOmega =  omega_expert;
    g_iMax_episode_len = 100;
    D_expert = inverted_pendulum_simulator( m_exp[i] );
    unsigned int nb_samples_exp = g_iNb_samples;
    g_iMax_episode_len = 100;
    gsl_matrix* omega_lstd =
      proj_lstd_lspi_ANIRL( D_expert, D_expert );
    g_mOmega = omega_lstd;
    g_iNb_samples = 0;
    g_iMax_episode_len = 3000;
    gsl_matrix* discard = inverted_pendulum_simulator( 100 );
    gsl_matrix_free( discard );
    unsigned int mean_control_steps = g_iNb_samples/100;
    gsl_matrix_free( omega_lstd );
    printf("1-LSTD %d %lf %lf %lf %lf %d\n", nb_samples_exp,
  	   g_dBest_t, g_dBest_error,
  	   g_dBest_true_error, g_dBest_diff, mean_control_steps );
    g_iMax_episode_len = 100;
    omega_imitation =
      proj_mc_lspi_ANIRL( D_expert, D, 2000 );
    g_mOmega = omega_imitation;
    g_iNb_samples = 0;
    g_iMax_episode_len = 3000;
    discard = inverted_pendulum_simulator( 100 );
    gsl_matrix_free( discard );
    mean_control_steps = g_iNb_samples/100;
    gsl_matrix_free( omega_imitation );
    gsl_matrix_free( D_expert );
    printf("1-MC %d %lf %lf %lf %lf %d\n", nb_samples_exp,
  	   g_dBest_t, g_dBest_error,
  	   g_dBest_true_error, g_dBest_diff, mean_control_steps);
  }


  //La seconde courbe montre qu'avec une matrice D, 
  //quelque soit la taille de la matrice D_E,
  //les deux variantes fonctionnent
  for( int i=0; i<3 ; i++ ){
    g_iNb_samples = 0;
    g_mOmega =  omega_expert;
    g_iMax_episode_len = 100;
    D_expert = inverted_pendulum_simulator( m_exp[i] );
    unsigned int nb_samples_exp = g_iNb_samples;
    g_iMax_episode_len = 100;
    gsl_matrix* omega_lstd =
      proj_lstd_lspi_ANIRL( D_expert, D );
    g_mOmega = omega_lstd;
    g_iNb_samples = 0;
    g_iMax_episode_len = 3000;
    gsl_matrix* discard = inverted_pendulum_simulator( 100 );
    gsl_matrix_free( discard );
    unsigned int mean_control_steps = g_iNb_samples/100;
    gsl_matrix_free( omega_lstd );
    printf("2-LSTD %d %lf %lf %lf %lf %d\n", nb_samples_exp,
  	   g_dBest_t, g_dBest_error,
  	   g_dBest_true_error, g_dBest_diff, mean_control_steps );
    g_iMax_episode_len = 100;
    omega_imitation =
      proj_mc_lspi_ANIRL( D_expert, D, 2000 );
    g_mOmega = omega_imitation;
    g_iNb_samples = 0;
    g_iMax_episode_len = 3000;
    discard = inverted_pendulum_simulator( 100 );
    gsl_matrix_free( discard );
    mean_control_steps = g_iNb_samples/100;
    gsl_matrix_free( omega_imitation );
    gsl_matrix_free( D_expert );
    printf("2-MC %d %lf %lf %lf %lf %d\n", nb_samples_exp,
  	   g_dBest_t, g_dBest_error,
  	   g_dBest_true_error, g_dBest_diff, mean_control_steps);
  }


  /* gsl_matrix* omega_imitation;*/
  /* unsigned int M = 1001; //To differentiate from above when greping. */
  /* //See the Makefile */
  /* int m_exp[] = {1,10,100}; */
  /* for( int i=0; i<3 ; i++ ){ */
  /*   g_iNb_samples = 0; */
  /*   g_mOmega =  omega_expert; */
  /*   g_iMax_episode_len = 30; */
  /*   D_expert = inverted_pendulum_simulator( m_exp[i] ); */
  /*   unsigned int nb_samples_exp = g_iNb_samples; */
  /*   g_iMax_episode_len = 100; */
  /*   gsl_matrix* omega_lstd = */
  /*     proj_lstd_lspi_ANIRL( D_expert, D ); */
  /*   g_mOmega = omega_lstd; */
  /*   g_iNb_samples = 0; */
  /*   g_iMax_episode_len = 3000; */
  /*   gsl_matrix* discard = inverted_pendulum_simulator( 100 ); */
  /*   gsl_matrix_free( discard ); */
  /*   unsigned int mean_control_steps = g_iNb_samples/100; */
  /*   gsl_matrix_free( omega_lstd ); */
  /*   printf("B %d %lf %lf %lf %lf %d\n", nb_samples_exp, */
  /* 	   g_dBest_t, g_dBest_error, */
  /* 	   g_dBest_true_error, g_dBest_diff, mean_control_steps ); */
  /*   g_iMax_episode_len = 100; */
  /*   omega_imitation = */
  /*     proj_mc_lspi_ANIRL( D_expert, D, M ); */
  /*   g_mOmega = omega_imitation; */
  /*   g_iNb_samples = 0; */
  /*   g_iMax_episode_len = 3000; */
  /*   discard = inverted_pendulum_simulator( 100 ); */
  /*   gsl_matrix_free( discard ); */
  /*   mean_control_steps = g_iNb_samples/100; */
  /*   gsl_matrix_free( omega_imitation ); */
  /*   gsl_matrix_free( D_expert ); */
  /*   printf("C %d %lf %lf %lf %lf %d\n", nb_samples_exp, */
  /* 	   g_dBest_t, g_dBest_error, */
  /* 	   g_dBest_true_error, g_dBest_diff, mean_control_steps); */
  /* } */
  gsl_matrix_free( g_mActions );
  gsl_matrix_free( omega_expert );
  expert_free();
  gsl_matrix_free( D );
  gsl_matrix_free( omega_0 );
  return 0;
}
