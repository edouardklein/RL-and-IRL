
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <math.h>
#include "RL_Globals.h"
#include "IRL_Globals.h"
#include "abbeel2004apprenticeship.h"
#include "utils.h"
#include "criteria.h"

#define PERFECT_MC_LENGTH 2000

gsl_matrix* g_mOmega_E = NULL;

gsl_matrix* g_mMu_E = NULL;

double g_dV_E = -1;

double true_diff_norm( gsl_matrix* omega ){
  g_mOmega = omega;
  unsigned int nb_samples_backup = g_iNb_samples;
  gsl_matrix* D = g_fSimulator( PERFECT_MC_LENGTH );
  g_iNb_samples = nb_samples_backup;
  gsl_matrix* mu = monte_carlo_mu( D );
  double norm = diff_norm( mu, g_mMu_E );
  gsl_matrix_free( mu );
  gsl_matrix_free( D );
  return norm;
}

void expert_just_set(){
  unsigned int nb_samples_backup = g_iNb_samples;
  g_mOmega = g_mOmega_E;
  gsl_matrix* D = g_fSimulator( PERFECT_MC_LENGTH );
  g_iNb_samples = nb_samples_backup;
  g_mMu_E = monte_carlo_mu( D );
  g_dV_E = value_func( D );
  gsl_matrix_free( D );
}

void expert_free(){
  gsl_matrix_free( g_mMu_E );
}

double value_func( gsl_matrix* D ){
  double answer = 0;
  double gamma_t = 1;
  unsigned int nb_episodes = 0;
  for( unsigned int i = 0; i<D->size1; i++ ){
    double r = gsl_matrix_get( D, i, D->size2-2 );
    double eoe = gsl_matrix_get( D, i, D->size2-1 );
    answer += gamma_t*r;
    if( (int)eoe == 1 ){
      gamma_t *= g_dGamma_lstdq;
    }else{
      gamma_t = 1;
      nb_episodes++;
    }
  }
  return answer/(double)nb_episodes;
}

double true_V_diff( gsl_matrix* omega ){
  g_mOmega = omega;
  unsigned int nb_samples_backup = g_iNb_samples;
  gsl_matrix* D = g_fSimulator( PERFECT_MC_LENGTH );
  g_iNb_samples = nb_samples_backup;
  double V_pi = value_func( D );
  double answer = fabs(g_dV_E - V_pi);
  gsl_matrix_free( D );
  return answer;  
}
