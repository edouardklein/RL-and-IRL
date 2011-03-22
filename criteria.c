#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <math.h>
#include "RL_Globals.h"
#include "IRL_Globals.h"
#include "abbeel2004apprenticeship.h"
#include "utils.h"

#define PERFECT_MC_LENGTH 1000

/* Public global where the omega of the expert is stored */
gsl_matrix* g_mOmega_E = NULL;

/* Private variable where the feature expectation 
   of the expert is stored*/
gsl_matrix* g_mMu_E = NULL;

/* Private variable where V_E(s_0) is stored */
double g_dV_E = -1;

/* "True" difference between the feature expectation
   of the expert and the one for the current policy :
   ||\mu_E-\mu||_2
   Computed via a big monte-carlo.
   expert_just_set() must be called before this function.
*/
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

/* Compute V(s0) dor the given set of trajectories */
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

/* Must call this one after the global variable
   g_mOmega_E is set, so that its feature expectation
   can be computed
*/
void expert_just_set(){
  unsigned int nb_samples_backup = g_iNb_samples;
  g_mOmega = g_mOmega_E;
  gsl_matrix* D = g_fSimulator( PERFECT_MC_LENGTH );
  g_iNb_samples = nb_samples_backup;
  g_mMu_E = monte_carlo_mu( D );
  g_dV_E = value_func( D );
  gsl_matrix_free( D );
}


/* |V^E(s_0)-V^\pi(s_0)| for the current policy.
   Objective measurement for task transfer.
   Computed via a big monte_carlo.
*/
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
