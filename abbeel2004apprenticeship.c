
#include <gsl/gsl_matrix.h>
#include <math.h>
#include <gsl/gsl_blas.h>
#include "RL_Globals.h"
#include "IRL_Globals.h"
#include "utils.h"
#include "criteria.h"
#include "LSPI.h"

double g_dBest_error;
double g_dBest_true_error;
double g_dBest_diff;
double g_dBest_t;
gsl_matrix* g_mBest_omega;

gsl_matrix* monte_carlo_mu( gsl_matrix* D ){
  gsl_matrix* answer = gsl_matrix_calloc( g_iP, 1 );
  double t = 0;
  unsigned int nb_traj = 0;
  gsl_matrix_view EOEs = 
    gsl_matrix_submatrix( D, 0, D->size2-1, D->size1, 1 ); 
  for( unsigned int i=0; i < D->size1 ; i++ ){
    gsl_matrix_view s = gsl_matrix_submatrix( D, i, 0, 
					      1, g_iS );
    gsl_matrix* delta_mu = g_fPsi( &s.matrix );
    double gamma_t = pow( g_dGamma_anirl, t );
    gsl_matrix_scale( delta_mu, gamma_t );
    gsl_matrix_add( answer, delta_mu );
    gsl_matrix_free( delta_mu );
    int eoe = (int)gsl_matrix_get( &EOEs.matrix, i, 0 );
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

gsl_matrix* proj_mc_lspi_ANIRL( gsl_matrix* D_E,
				gsl_matrix* D,
				unsigned int m ){
  gsl_matrix* omega_0 = gsl_matrix_calloc( g_iK, 1 );

  gsl_matrix* omega = gsl_matrix_calloc( g_iK, 1 );

  g_mOmega = omega;
  gsl_matrix* trans = g_fSimulator( m );

  gsl_matrix* mu = monte_carlo_mu( trans );
  gsl_matrix_free( trans );

  gsl_matrix* mu_E = monte_carlo_mu( D_E );

  gsl_matrix* theta = gsl_matrix_alloc(g_iP, 1 );
  gsl_matrix_memcpy( theta, mu_E );
  gsl_matrix_sub( theta, mu );
  gsl_vector_view theta_v = gsl_matrix_column( theta, 0 );
  double theta_norm = gsl_blas_dnrm2( &theta_v.vector );
  if( theta_norm != 0 )
    gsl_matrix_scale( theta, 1./theta_norm );

     gsl_matrix* bar_mu = gsl_matrix_alloc( g_iP, 1 );
     gsl_matrix_memcpy( bar_mu, mu );

	double t = diff_norm( mu_E, bar_mu );
	unsigned int nb_it = 0;
	g_dBest_error = diff_norm( mu_E, mu );
	g_dBest_true_error = true_diff_norm( omega );
	g_dBest_diff = true_V_diff( omega );
	g_dBest_t = t;
	g_mBest_omega = gsl_matrix_alloc( omega->size1, omega->size2 );
	gsl_matrix_memcpy( g_mBest_omega, omega );

   while( t > g_dEpsilon_anirl && nb_it < g_iIt_max_anirl ){

    double empirical_err = diff_norm( mu_E, mu );
    double true_err = true_diff_norm( omega );
    double true_V = true_V_diff( omega );
    printf( "%d %d %lf %lf %lf %lf\n", 
	    m, nb_it,
	    t, empirical_err, true_err, true_V );
           if( true_err <= g_dBest_true_error ){
      g_dBest_error = empirical_err;
      g_dBest_true_error = true_err;
      g_dBest_diff = true_V;
      g_dBest_t = t;
      gsl_matrix_memcpy( g_mBest_omega, omega );
    }

    for( unsigned int i = 0 ; i < D->size1 ; i++ ){
      gsl_matrix_view state = 
	gsl_matrix_submatrix( D, i, 0, 1, g_iS );
      gsl_matrix* psi_s = g_fPsi( &state.matrix );
      gsl_matrix_view r = 
	gsl_matrix_submatrix( D, i, 2*g_iS+g_iA, 1, 1 );
      gsl_blas_dgemm( CblasTrans, CblasNoTrans, 1.0, 
		       theta, psi_s, 0.0, &r.matrix );
      gsl_matrix_free( psi_s );
    }

    gsl_matrix_free( omega );
    omega = lspi( D, omega_0 );

    g_mOmega = omega;
    trans = g_fSimulator( m );

    gsl_matrix_free( mu );
    mu = monte_carlo_mu( trans );
    gsl_matrix_free( trans );

    gsl_matrix* mu_barmu = 
      gsl_matrix_alloc( g_iP, 1);
    gsl_matrix* muE_barmu = 
      gsl_matrix_alloc( g_iP, 1);
    gsl_matrix* num = gsl_matrix_alloc( 1, 1 );
    gsl_matrix* denom = gsl_matrix_alloc( 1, 1 );
    gsl_matrix* delta_bar_mu = 
      gsl_matrix_alloc( g_iP, 1);
    gsl_matrix_memcpy( mu_barmu, mu );
    gsl_matrix_sub( mu_barmu, bar_mu );
    gsl_matrix_memcpy( muE_barmu, mu_E );
    gsl_matrix_sub( muE_barmu, bar_mu );
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

    gsl_matrix_memcpy( theta, mu_E );
    gsl_matrix_sub( theta, bar_mu );
    theta_v = gsl_matrix_column( theta, 0 );
    theta_norm = gsl_blas_dnrm2( &theta_v.vector );
    if( theta_norm != 0 )
      gsl_matrix_scale( theta, 1./theta_norm );

    t = diff_norm( mu_E, bar_mu );
    nb_it++;
  }

  double empirical_err = diff_norm( mu_E, mu );
  double true_err = true_diff_norm( omega );
  double true_V = true_V_diff( omega );
  printf( "%d %d %lf %lf %lf %lf\n", 
	  m, nb_it, 
	  t, empirical_err, true_err, true_V );
  if( true_err <= g_dBest_true_error ){
    g_dBest_error = empirical_err;
    g_dBest_true_error = true_err;
    g_dBest_diff = true_V;
    g_dBest_t = t;
    gsl_matrix_memcpy( g_mBest_omega, omega );
  }
  gsl_matrix_free( omega_0 );
  gsl_matrix_free( mu );
  gsl_matrix_free( mu_E );
  gsl_matrix_free( bar_mu );
  gsl_matrix_free( theta );
  gsl_matrix_free( omega );
  return g_mBest_omega;
}
