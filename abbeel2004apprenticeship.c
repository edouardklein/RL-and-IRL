#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <math.h>
#include "greedy.h"
#include "LSPI.h"
#include "utils.h"
#include "GridWorld_simulator.h"
#define NB_IT_MAX 40
/* Compute the mean reward over nb_ep episodes */
double quality( gsl_matrix* (*simulator)(int), 
		unsigned int s, unsigned int a, 
		gsl_matrix* omega, unsigned int nb_ep ){
  g_mOmega = gsl_matrix_alloc( omega->size1, 1 );
  gsl_matrix_memcpy( g_mOmega, omega );
  g_mActions = file2matrix( ACTION_FILE, g_iA );
  unsigned int nb_samples_backup = g_iNb_samples;
  gsl_matrix* D = simulator( nb_ep );
  g_iNb_samples = nb_samples_backup;
  gsl_matrix_free( g_mOmega );
  gsl_matrix_free( g_mActions );
  double cumulated_reward = 0;
  for( unsigned int i = 0 ; i < D->size1 ; i++ ){
    cumulated_reward += gsl_matrix_get( D, i, 2*s+a );
  }
  double quality = cumulated_reward / (double)(D->size1);
  gsl_matrix_free( D );
  return quality;
}

/* Compute an estimate of \mu using the monte carlo method,
   given a set of trajectories 
   \hat\mu = {1\over m }\sum_{i=0}^m
   \sum_{t=0}^\infty\gamma^t\psi(s_t^{(i)})
*/
gsl_matrix* monte_carlo_mu( gsl_matrix* states,gsl_matrix* EOEs,
			    double gamma,
			    gsl_matrix* (*psi)(gsl_matrix*)){
  gsl_matrix_view s = 
    gsl_matrix_submatrix( states, 0, 0, 
			  1, states->size2 );
  gsl_matrix* first_psi = psi( &s.matrix );
  gsl_matrix* answer = gsl_matrix_calloc( first_psi->size1,
					  first_psi->size2 );
  gsl_matrix_free( first_psi );
  double t = 0;
  unsigned int nb_traj = 0;
  for( unsigned int i=0; i < states->size1 ; i++ ){
    s = gsl_matrix_submatrix( states, i, 0, 
			      1, states->size2 );
    gsl_matrix* delta_mu = psi( &s.matrix );
    double gamma_t = pow( gamma, t );
    gsl_matrix_scale( delta_mu, gamma_t );
    gsl_matrix_add( answer, delta_mu );
    gsl_matrix_free( delta_mu );
    int eoe = (int)gsl_matrix_get( EOEs, i, 0 );
    if( eoe == 0 ){ //End of episode
      t = 0;
      nb_traj++;
    }else{
      t++;
    }
  }
  gsl_matrix_scale( answer, 1./(double)nb_traj );
  return answer;
}

/* Abbeel and Ng's IRL algorithm (ANIRL), with the projection 
   method, monte-carlo estimation and LSPI as the MDP solver.
   Given a simulator and an expert's trace, returns the \omega 
   matrix which defines the optimal policy as found by LSPI
   under the reward R = \theta^T\psi. 
   Note that \omega \equiv \pi
*/
gsl_matrix* proj_mc_lspi_ANIRL( gsl_matrix* expert_trans,
				gsl_matrix* (*simulator)(int),
				gsl_matrix* D,
				unsigned int s, unsigned int a,
				unsigned int k, unsigned int m,
				double gamma, 
				double gamma_lspi,
				double epsilon,
				double epsilon_lspi,
				gsl_matrix* (*phi)(gsl_matrix*),
				gsl_matrix* (*psi)(gsl_matrix*)){
  double max_q = 0;
  gsl_matrix* omega_0 = gsl_matrix_calloc( k, 1 );
  /* \omega \leftarrow 0 */
  gsl_matrix* omega = gsl_matrix_calloc( k, 1 );
  /* D_\pi \leftarrow simulator( m, \omega ) */
  g_mOmega = gsl_matrix_calloc( k, 1 );
  gsl_matrix_memcpy( g_mOmega, omega );
  g_mActions = file2matrix( ACTION_FILE, g_iA );
  gsl_matrix* trans = simulator( m );
  gsl_matrix_free( g_mOmega );
  gsl_matrix_free( g_mActions );
  /* \mu \leftarrow mc( D_\pi, \gamma, \psi ) */
  gsl_matrix_view states = 
    gsl_matrix_submatrix( trans, 0, 0, trans->size1, s );
  gsl_matrix_view EOEs = 
    gsl_matrix_submatrix( trans, 0, 2*s+a+1, trans->size1, 1 );
  gsl_matrix* mu = 
    monte_carlo_mu( &states.matrix, &EOEs.matrix, gamma, psi );
  gsl_matrix_free( trans );
  /* \mu_E \leftarrow mc( D_E, \gamma, \psi ) */
  states = gsl_matrix_submatrix( expert_trans, 0, 0, 
				 expert_trans->size1, s );
  EOEs = gsl_matrix_submatrix( expert_trans, 0, 2*s+a+1,
			       expert_trans->size1, 1 );
  gsl_matrix* mu_E = 
    monte_carlo_mu( &states.matrix, &EOEs.matrix, gamma, psi );
  /* \theta \leftarrow {\mu_E - \mu\over ||\mu_E - \mu||_2} */
  gsl_matrix* theta = gsl_matrix_alloc(mu->size1, mu->size2 );
  gsl_matrix_memcpy( theta, mu_E );
  gsl_matrix_sub( theta, mu );
  gsl_vector_view theta_v = gsl_matrix_column( theta, 0 );
  double theta_norm = gsl_blas_dnrm2( &theta_v.vector );
  if( theta_norm != 0 )
    gsl_matrix_scale( theta, 1./theta_norm );
  /* \bar\mu \leftarrow \mu*/
  gsl_matrix* bar_mu = gsl_matrix_alloc( mu->size1, mu->size2 );
  gsl_matrix_memcpy( bar_mu, mu );
  /* t \leftarrow ||\mu_E - \bar\mu||_2*/
  gsl_vector_view barmu_v = gsl_matrix_column( bar_mu, 0 );
  gsl_vector_view muE_v = gsl_matrix_column( mu_E, 0 );
  gsl_vector* diff = gsl_vector_calloc( mu->size1 );
  gsl_vector_memcpy( diff, &barmu_v.vector );
  gsl_vector_sub( diff, &muE_v.vector );
  double t = gsl_blas_dnrm2( diff );
  unsigned int nb_it = 0;
  while( t > epsilon && nb_it < NB_IT_MAX ){
    /**/
    double q = quality( gridworld_simulator, s, a, 
			omega, 1000 );
    if( q > max_q ){
      max_q = q;
    }
    /**/
    printf("%d %d %d %lf\n", m, nb_it, g_iNb_samples, t );
    /* D.r \leftarrow \theta^T\psi(D.s) */
    for( unsigned int i = 0 ; i < D->size1 ; i++ ){
      gsl_matrix_view state = 
	gsl_matrix_submatrix( D, i, 0, 1, s );
      gsl_matrix* psi_s = psi( &state.matrix );
      gsl_matrix_view r = 
	gsl_matrix_submatrix( D, i, 2*s+a, 1, 1 );
      gsl_blas_dgemm( CblasTrans, CblasNoTrans, 1.0, 
		       theta, psi_s, 0.0, &r.matrix );
      gsl_matrix_free( psi_s );
    }
    /* \omega \leftarrow LSPI(D,k,\phi,\gamma_{LSPI},
       \epsilon_{LSPI}, \omega_0)*/
    gsl_matrix_free( omega );
    omega = lspi( D, k, s, a, phi, gamma_lspi, epsilon_lspi, 
		  omega_0 );
    /* D_\pi \leftarrow simulator( m, \omega ) */
    g_mOmega = gsl_matrix_calloc( k, 1 );
    gsl_matrix_memcpy( g_mOmega, omega );
    g_mActions = file2matrix( ACTION_FILE, g_iA );
    trans = simulator( m );
    gsl_matrix_free( g_mOmega );
    gsl_matrix_free( g_mActions );
    /* \mu \leftarrow mc( D_\pi, \gamma, \psi ) */
    gsl_matrix_free( mu );
    states = 
      gsl_matrix_submatrix( trans, 0, 0, trans->size1, s );
    EOEs = 
      gsl_matrix_submatrix( trans, 0, 2*s+a+1, trans->size1, 1 );
    mu = 
      monte_carlo_mu( &states.matrix, &EOEs.matrix,gamma, psi );
    gsl_matrix_free( trans );
    /* \bar\mu \leftarrow \bar\mu + 
     { (\mu-\bar\mu)^T (\mu_E-\bar\mu) \over 
     (\mu-\bar\mu)^T (\mu-\bar\mu) }
     (\mu-\bar\mu) */
    gsl_matrix* mu_barmu = 
      gsl_matrix_alloc( mu->size1, mu->size2);
    gsl_matrix* muE_barmu = 
      gsl_matrix_alloc( mu->size1, mu->size2);
    gsl_matrix* num = gsl_matrix_alloc( 1, 1 );
    gsl_matrix* denom = gsl_matrix_alloc( 1, 1 );
    gsl_matrix* delta_bar_mu = 
      gsl_matrix_alloc( mu->size1, mu->size2);
    gsl_matrix_memcpy( mu_barmu, mu );
    gsl_matrix_sub( mu_barmu, bar_mu );
    gsl_matrix_memcpy( muE_barmu, mu_E );
    gsl_matrix_sub( muE_barmu, bar_mu ); //Check here
    gsl_blas_dgemm( CblasTrans, CblasNoTrans, 1.0,
		    mu_barmu, muE_barmu, 0.0, num );
    gsl_blas_dgemm( CblasTrans, CblasNoTrans, 1.0,
		    mu_barmu, mu_barmu, 0.0, denom );
    gsl_matrix_memcpy( delta_bar_mu, mu_barmu );
    double scale = gsl_matrix_get( num, 0, 0 ) / 
      gsl_matrix_get( denom, 0, 0 );
    gsl_matrix_scale( delta_bar_mu, scale );
    gsl_matrix_add( bar_mu, delta_bar_mu );
    gsl_matrix_free( num );
    gsl_matrix_free( denom );
    gsl_matrix_free( mu_barmu );
    gsl_matrix_free( muE_barmu );
    gsl_matrix_free( delta_bar_mu );
    /* \theta \leftarrow 
       {\mu_E - \bar\mu\over ||\mu_E - \bar\mu||_2} */
    gsl_matrix_memcpy( theta, mu_E );
    gsl_matrix_sub( theta, bar_mu );
    theta_v = gsl_matrix_column( theta, 0 );
    theta_norm = gsl_blas_dnrm2( &theta_v.vector );
    if( theta_norm != 0 )
      gsl_matrix_scale( theta, 1./theta_norm );
    /* t\leftarrow ||\mu_E - \bar\mu||_2 */
    gsl_vector_memcpy( diff, &barmu_v.vector );
    gsl_vector_sub( diff, &muE_v.vector );
    t = gsl_blas_dnrm2( diff );
    nb_it++;
  }
  double q = quality( gridworld_simulator, s, a, 
		      omega, 1000 );
  if( q > max_q ){
    max_q = q;
  }
  printf("%d %d %d %lf\n", m, nb_it, g_iNb_samples, t );
  printf("AN MAXQ %d %lf\n", m, max_q);
  gsl_matrix_free( omega_0 );
  gsl_matrix_free( mu );
  gsl_matrix_free( mu_E );
  gsl_matrix_free( bar_mu );
  gsl_vector_free( diff );
  gsl_matrix_free( theta );
  return omega;
}
				
