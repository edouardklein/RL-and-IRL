#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <math.h>
#include "greedy.h"
#include "utils.h"
#include "criteria.h"
#include "LSPI.h"
#include "RL_Globals.h"
#include "IRL_Globals.h"
/* #include <math.h> */
/* #include "GridWorld_simulator.h" */


/* Compute an estimate of \mu(s_0) using the LSTD\mu algorithm,
   in its on policy form. Typically used for the expert.
   The D_mu matrix has the form
   s a s' psi(s) eoe
   s_0 is taken from the first transition of D_\mu
*/
gsl_matrix* lstd_mu_op(  gsl_matrix* D_mu ){
  gsl_matrix* s_0 = gsl_matrix_alloc( 1, g_iS );
  gsl_matrix_view s_0_src = gsl_matrix_submatrix( D_mu, 0, 0,
						  1, g_iS );
  gsl_matrix_memcpy( s_0, &s_0_src.matrix );
  // \tilde A \leftarrow 0
  gsl_matrix* A = gsl_matrix_calloc( g_iP, g_iP );
  // \tilde b \leftarrow 0
  gsl_matrix* b = gsl_matrix_calloc( g_iP, g_iP );
  // for each (s,a,s',\psi(s),eoe) \in D_mu
  for( unsigned int i=0; i < D_mu->size1 ; i++ ){
    // \tilde A \leftarrow \tilde A + \psi(s)\left(\psi(s) 
    // - \gamma \psi(s')\right)^T
    gsl_matrix_view psi_s = 
      gsl_matrix_submatrix( D_mu, i, g_iS+g_iA+g_iS, 1, g_iP );
    gsl_matrix_view s_dash = 
      gsl_matrix_submatrix( D_mu, i, g_iS+g_iA, 1, g_iS);
    gsl_matrix* psi_dash = g_fPsi( &s_dash.matrix );
    gsl_matrix_scale( psi_dash, g_dGamma_anirl );
    double eoe = gsl_matrix_get( D_mu, i, g_iS+g_iA+g_iS+g_iP );
    gsl_matrix_scale( psi_dash, eoe );
    gsl_matrix* delta_psi = gsl_matrix_calloc( g_iP, 1 );
    gsl_matrix_transpose_memcpy( delta_psi, &psi_s.matrix );
    gsl_matrix_sub( delta_psi, psi_dash );
    gsl_matrix* deltaA = gsl_matrix_calloc( g_iP, g_iP );
    gsl_blas_dgemm( CblasTrans, CblasTrans, 1., 
		      &psi_s.matrix, delta_psi, 0., deltaA);
    gsl_matrix_add( A, deltaA );
    //\tilde b \leftarrow \tilde b + \psi(s)\psi(s)^T
    gsl_matrix* delta_b = gsl_matrix_alloc( g_iP, g_iP );
    /* \psi(s) is in line in the code but in column in the 
     comments */
    gsl_blas_dgemm( CblasTrans, CblasNoTrans, 1.0,
		    &psi_s.matrix, &psi_s.matrix, 0.0, delta_b );
    gsl_matrix_add( b, delta_b );
    gsl_matrix_free( deltaA );
    gsl_matrix_free( delta_psi );
    gsl_matrix_free( psi_dash );
    gsl_matrix_free( delta_b );
  }
  /*\tilde \omega^\pi \leftarrow 
    (\tildeA + \lambda Id) ^{-1}\tilde b */
  gsl_matrix* lambdaI = gsl_matrix_alloc( A->size1, A->size2 );
  gsl_matrix_set_identity( lambdaI );
  gsl_matrix_scale( lambdaI, g_dLambda_lstdmu );
  gsl_matrix_add( A, lambdaI );
  gsl_matrix_free( lambdaI );
  gsl_matrix* omega_pi = gsl_matrix_alloc( g_iP, g_iP );
  gsl_permutation* p = gsl_permutation_alloc( g_iP );
  int signum;
  gsl_linalg_LU_decomp( A, p, &signum );
  for( unsigned int i = 0 ; i < g_iP ; i++ ){
    gsl_vector_view b_v = gsl_matrix_column( b, i );
    gsl_vector_view o_v = gsl_matrix_column( omega_pi, i );
    gsl_linalg_LU_solve( A, p, &b_v.vector, &o_v.vector );
  }
  gsl_permutation_free( p );
  gsl_matrix_free( A );
  gsl_matrix_free( b );
   //\mu_\pi(s) \leftarrow \tilde\omega_\pi^T\psi(s)
  gsl_matrix* psi_s_0 = g_fPsi( s_0 );
  gsl_matrix* mu = gsl_matrix_alloc( g_iP, 1 );
  gsl_blas_dgemm( CblasTrans, CblasNoTrans, 1.0,
		  omega_pi, psi_s_0, 0.0, mu );
  gsl_matrix_free( omega_pi );
  gsl_matrix_free( psi_s_0 );
  gsl_matrix_free( s_0 );
  return mu;
}

/* 
   Compute an estimate of \mu(s_0) using the LSTD\mu algorithm
   The D_mu matrix has the form
   s a s' psi(s) eoe
*/
gsl_matrix* lstd_mu(  gsl_matrix* D_mu,
		      gsl_matrix* (*pi)(gsl_matrix*), 
		      gsl_matrix* s_0 ){
  // \tilde A \leftarrow 0
  gsl_matrix* A = gsl_matrix_calloc( g_iK, g_iK );
  // \tilde b \leftarrow 0
  gsl_matrix* b = gsl_matrix_calloc( g_iK, g_iP );
  // for each (s,a,s',\psi(s),eoe) \in D_mu
  for( unsigned int i=0; i < D_mu->size1 ; i++ ){
    // \tilde A \leftarrow \tilde A + \phi(s,a)\left(\phi(s,a) 
    // - \gamma \phi(s',\pi(s'))\right)^T
    gsl_matrix_view sa = 
      gsl_matrix_submatrix( D_mu, i, 0, 1, g_iS+g_iA);
    gsl_matrix* phi_sa = g_fPhi( &sa.matrix );
    gsl_matrix* sa_dash = gsl_matrix_calloc( 1, g_iS+g_iA );
    gsl_matrix_view sdash_dst = gsl_matrix_submatrix( sa_dash, 
						      0, 0,
						      1, g_iS );
    gsl_matrix_view sdash_src = 
      gsl_matrix_submatrix( D_mu, i, g_iS+g_iA, 1, g_iS);
    gsl_matrix_memcpy( &sdash_dst.matrix, &sdash_src.matrix );
    gsl_matrix_view adash_dst = 
      gsl_matrix_submatrix( sa_dash, 0, g_iS, 1, g_iA );
    gsl_matrix* adash_src = pi( &sdash_src.matrix );
    gsl_matrix_memcpy( &adash_dst.matrix, adash_src );
    gsl_matrix* phi_dash = g_fPhi( sa_dash );
    gsl_matrix_scale( phi_dash, g_dGamma_anirl );
    double eoe = gsl_matrix_get( D_mu, i, g_iS+g_iA+g_iS+g_iP );
    gsl_matrix_scale( phi_dash, eoe );
    gsl_matrix* delta_phi = gsl_matrix_calloc( g_iK, 1 );
    gsl_matrix_memcpy( delta_phi, phi_sa );
    gsl_matrix_sub( delta_phi, phi_dash );
    gsl_matrix* deltaA = gsl_matrix_calloc( g_iK, g_iK );
    gsl_blas_dgemm( CblasNoTrans, CblasTrans, 1., 
		      phi_sa, delta_phi, 0., deltaA);
    gsl_matrix_add( A, deltaA );
    //\tilde b \leftarrow \tilde b + \phi(s,a)\psi(s)^T
    gsl_matrix_view psi_s = 
      gsl_matrix_submatrix( D_mu, i, g_iS+g_iA+g_iS, 1, g_iP );
    gsl_matrix* delta_b = gsl_matrix_alloc( g_iK, g_iP );
    /*\psi(s) is in line in the code but in column 
      in the comments*/ 
    gsl_blas_dgemm( CblasNoTrans, CblasNoTrans, 1.0,
		    phi_sa, &psi_s.matrix, 0.0, delta_b );
    gsl_matrix_add( b, delta_b );
    gsl_matrix_free( deltaA );
    gsl_matrix_free( delta_phi );
    gsl_matrix_free( phi_dash );
    gsl_matrix_free( adash_src );
    gsl_matrix_free( sa_dash );
    gsl_matrix_free( phi_sa );
    gsl_matrix_free( delta_b );
  }
  /*\tilde \omega^\pi \leftarrow 
    (\tildeA + \lambda Id) ^{-1}\tilde b */
  gsl_matrix* lambdaI = gsl_matrix_alloc( A->size1, A->size2 );
  gsl_matrix_set_identity( lambdaI );
  gsl_matrix_scale( lambdaI, g_dLambda_lstdmu );
  gsl_matrix_add( A, lambdaI );
  gsl_matrix_free( lambdaI );
  gsl_matrix* omega_pi = gsl_matrix_alloc( g_iK, g_iP );
  gsl_permutation* p = gsl_permutation_alloc( g_iK );
  int signum;
  gsl_linalg_LU_decomp( A, p, &signum );
  for( unsigned int i = 0 ; i < g_iP ; i++ ){
    gsl_vector_view b_v = gsl_matrix_column( b, i );
    gsl_vector_view o_v = gsl_matrix_column( omega_pi, i );
    gsl_linalg_LU_solve( A, p, &b_v.vector, &o_v.vector );
  }
  gsl_permutation_free( p );
  gsl_matrix_free( A );
  gsl_matrix_free( b );
  //\mu_\pi(s) \leftarrow \tilde\omega_\pi^T\phi(s,\pi(s))
  gsl_matrix* s_pi_s = gsl_matrix_alloc( 1, g_iS+g_iA );
  gsl_matrix_view s_dst = gsl_matrix_submatrix( s_pi_s, 
						0, 0,
						1, g_iS);
  gsl_matrix_memcpy( &s_dst.matrix, s_0 );
  gsl_matrix_view pi_s_dst = gsl_matrix_submatrix( s_pi_s,
						    0, g_iS, 
						    1, g_iA);
  gsl_matrix* pi_s_src = pi( s_0 );
  gsl_matrix_memcpy( &pi_s_dst.matrix, pi_s_src );
  gsl_matrix* phi_s_pi_s = g_fPhi( s_pi_s );
  gsl_matrix* mu = gsl_matrix_alloc( g_iP, 1 );
  gsl_blas_dgemm( CblasTrans, CblasNoTrans, 1.0,
		  omega_pi, phi_s_pi_s, 0.0, mu );
  gsl_matrix_free( omega_pi );
  gsl_matrix_free( s_pi_s );
  gsl_matrix_free( pi_s_src );
  gsl_matrix_free( phi_s_pi_s );
  return mu;
}

/* Abbeel and Ng's IRL algorithm (ANIRL), with the projection 
   method, LSTDMu estimation and LSPI as the MDP solver.
   Given an expert's trace, returns the \omega 
   matrix which defines the optimal policy as found by LSPI
   under the reward R = \theta^T\psi. 
   Note that \omega \equiv \pi
*/
gsl_matrix* proj_lstd_lspi_ANIRL( gsl_matrix* D_E, 
				  gsl_matrix* D ){
  gsl_matrix* s_0 = gsl_matrix_alloc( 1, g_iS );
  gsl_matrix_view s_0_src = gsl_matrix_submatrix( D_E,
						  0, 0,
						  1, g_iS );
  gsl_matrix_memcpy( s_0, &s_0_src.matrix );
  unsigned int m = 0; //0 is characteristic of LSTDMu when 
  //plotting
  gsl_matrix* omega_0 = gsl_matrix_calloc( g_iK, 1 );
  /* \omega \leftarrow 0 */
  gsl_matrix* omega = gsl_matrix_calloc( g_iK, 1 );
  /* D_\mu \leftarrow D 
     D_\mu.r \leftarrow \psi(D.s) */
  gsl_matrix* D_mu = gsl_matrix_alloc( D->size1, 
				       g_iS+g_iA+g_iS+g_iP+1 );
  gsl_matrix_view Dsas = 
    gsl_matrix_submatrix( D, 0, 0, D->size1, g_iS+g_iA+g_iS );
  gsl_matrix_view Dmusas = 
    gsl_matrix_submatrix( D_mu, 0, 0, D->size1, g_iS+g_iA+g_iS);
  gsl_matrix_memcpy( &Dmusas.matrix, &Dsas.matrix );
  for( unsigned int i = 0 ; i<D->size1 ; i++ ){
    gsl_matrix_view vS = gsl_matrix_submatrix( D, i, 0, 1,g_iS);
    gsl_matrix* psi_s_src = g_fPsi( &vS.matrix );
    gsl_matrix_view psi_s_dst = 
      gsl_matrix_submatrix( D_mu, i, g_iS+g_iA+g_iS, 1, g_iP );
    /* D.\psi(s) is in line in the code but in columns in
       the comments */
    gsl_matrix_transpose_memcpy( &psi_s_dst.matrix, psi_s_src );
    gsl_matrix_free( psi_s_src );
  }
  gsl_matrix_view Deoe = 
    gsl_matrix_submatrix( D, 0, g_iS+g_iA+g_iS+1, D->size1, 1 );
  gsl_matrix_view Dmueoe = 
    gsl_matrix_submatrix( D_mu, 0, g_iS+g_iA+g_iS+g_iP, 
			  D->size1, 1 ); 
  gsl_matrix_memcpy( &Dmueoe.matrix, &Deoe.matrix );
  /* \mu \leftarrow LSTD\mu( D_\mu, k, p, s, a, \phi,
     \psi, \gamma, \pi ) */
  g_mOmega = omega;
  gsl_matrix* mu = lstd_mu( D_mu, &greedy_policy, s_0 );
  /* D_E.r \leftarrow \psi(D_E.s) */
  gsl_matrix* D_E_mu = 
    gsl_matrix_alloc( D_E->size1, g_iS+g_iA+g_iS+g_iP+1 );
  gsl_matrix_view DEsas_dst = 
    gsl_matrix_submatrix( D_E_mu, 0, 0, 
			  D_E_mu->size1,g_iS+g_iA+g_iS );
  gsl_matrix_view DEsas_src = 
    gsl_matrix_submatrix(D_E, 0, 0, D_E->size1, g_iS+g_iA+g_iS);
  gsl_matrix_memcpy( &DEsas_dst.matrix, &DEsas_src.matrix );
  for( unsigned int i = 0 ; i<D_E_mu->size1 ; i++ ){
    gsl_matrix_view vS = 
      gsl_matrix_submatrix(D_E_mu, i, 0, 1, g_iS );
    gsl_matrix* psi_s_src = g_fPsi( &vS.matrix );
    gsl_matrix_view psi_s_dst = 
      gsl_matrix_submatrix( D_E_mu, i, g_iS+g_iA+g_iS, 1, g_iP);
    /* D.\psi(s) is in line in the code but in columns in
       the comments */
    gsl_matrix_transpose_memcpy( &psi_s_dst.matrix, psi_s_src );
    gsl_matrix_free( psi_s_src );
  }
  gsl_matrix_view DEeoe_dst = 
    gsl_matrix_submatrix( D_E_mu, 0, g_iS+g_iA+g_iS+g_iP,
			  D_E_mu->size1, 1 );
  gsl_matrix_view DEeoe_src = 
    gsl_matrix_submatrix( D_E, 0, g_iS+g_iA+g_iS+1, 
			  D_E->size1, 1 ); 
  gsl_matrix_memcpy( &DEeoe_dst.matrix, &DEeoe_src.matrix );
  /* \mu_E \leftarrow on-LSTD_\mu( D_E, k, p, s, a, \psi,
     \phi, \gamma) */
  gsl_matrix* mu_E = lstd_mu_op( D_E_mu );
  gsl_matrix_free( D_E_mu );
  /* \theta \leftarrow {\mu_E - \mu\over ||\mu_E - \mu||_2} */
  gsl_matrix* theta = gsl_matrix_alloc( g_iP, 1 );
  gsl_matrix_memcpy( theta, mu_E );
  gsl_matrix_sub( theta, mu );
  gsl_vector_view theta_v = gsl_matrix_column( theta, 0 );
  double theta_norm = gsl_blas_dnrm2( &theta_v.vector );
  if( theta_norm != 0 )
    gsl_matrix_scale( theta, 1./theta_norm );
  /* \bar\mu \leftarrow \mu*/
  gsl_matrix* bar_mu = gsl_matrix_alloc( g_iP, 1 );
  gsl_matrix_memcpy( bar_mu, mu );
  /* t \leftarrow ||\mu_E - \bar\mu||_2*/
  double t = diff_norm( mu_E, bar_mu );
  unsigned int nb_it = 0;
  g_dBest_error = diff_norm( mu_E, mu );
  g_dBest_true_error = true_diff_norm( omega );
  g_dBest_diff = true_V_diff( omega );
  g_dBest_t = t;
  g_mBest_omega = gsl_matrix_alloc( omega->size1, omega->size2 );
  gsl_matrix_memcpy( g_mBest_omega, omega );
  /* while t > \epsilon */
  while( t > g_dEpsilon_anirl && nb_it < g_iIt_max_anirl ){
    /* Output of the different criteria */
    double empirical_err = diff_norm( mu_E, mu );
    double true_err = true_diff_norm( omega );
    double true_V = true_V_diff( omega );
    printf( "%d %d %lf %lf %lf %lf\n", 
	    m, nb_it, 
	    t, empirical_err, true_err, true_V );
    //if( empirical_err <= g_dBest_error ){
    if( true_err <= g_dBest_true_error ){
      g_dBest_error = empirical_err;
      g_dBest_true_error = true_err;
      g_dBest_diff = true_V;
      g_dBest_t = t;
      gsl_matrix_memcpy( g_mBest_omega, omega );
    }
    /* D.r \leftarrow \theta^T\psi(D.s) */
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
    /* \omega \leftarrow LSPI(D,k,\phi,\gamma_{LSPI},
       \epsilon_{LSPI}, \omega_0)*/
    gsl_matrix_free( omega );
    omega = lspi( D, omega_0 );
    /* \mu \leftarrow LSTD_\mu( D_\mu, k, p, s, a, \phi,
       \psi, \gamma, \pi ) */
    g_mOmega = omega;
    gsl_matrix_free( mu );
    mu = lstd_mu( D_mu, &greedy_policy, s_0);
    /* \bar\mu \leftarrow \bar\mu + 
     { (\mu-\bar\mu)^T (\mu_E-\bar\mu) \over 
     (\mu-\bar\mu)^T (\mu-\bar\mu) }
     (\mu-\bar\mu) */
    gsl_matrix* mu_barmu = gsl_matrix_alloc( g_iP, 1 );
    gsl_matrix* muE_barmu = gsl_matrix_alloc( g_iP, 1 );
    gsl_matrix* num = gsl_matrix_alloc( 1, 1 );
    gsl_matrix* denom = gsl_matrix_alloc( 1, 1 );
    gsl_matrix* delta_bar_mu = gsl_matrix_alloc( g_iP, 1 );
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
    if( isnan( scale ) ){
      gsl_matrix_free( num );
      gsl_matrix_free( denom );
      gsl_matrix_free( mu_barmu );
      gsl_matrix_free( muE_barmu );
      gsl_matrix_free( delta_bar_mu );
      gsl_matrix_free( s_0 );
      gsl_matrix_free( D_mu );
      gsl_matrix_free( omega_0 );
      gsl_matrix_free( mu );
      gsl_matrix_free( mu_E );
      gsl_matrix_free( bar_mu );
      gsl_matrix_free( theta );
      fprintf(stderr,"lstd_ANIRL returning early because it's stuck\n");
      return omega;
    }
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
    t = diff_norm( mu_E, bar_mu );
    nb_it++;
  }
  /* Last output of the different criteria */
  double empirical_err = diff_norm( mu_E, mu );
  double true_err = true_diff_norm( omega );
  double true_V = true_V_diff( omega );
  printf( "%d %d %lf %lf %lf %lf\n", 
	  m, nb_it, 
	  t, empirical_err, true_err, true_V );
  //  if( empirical_err <= g_dBest_error ){
  if( true_err <= g_dBest_true_error ){
    g_dBest_error = empirical_err;
    g_dBest_true_error = true_err;
    g_dBest_diff = true_V;
    g_dBest_t = t;
    gsl_matrix_memcpy( g_mBest_omega, omega );
  }
  gsl_matrix_free( s_0 );
  gsl_matrix_free( D_mu );
  gsl_matrix_free( omega_0 );
  gsl_matrix_free( mu );
  gsl_matrix_free( mu_E );
  gsl_matrix_free( bar_mu );
  gsl_matrix_free( theta );
  gsl_matrix_free( omega );
  return g_mBest_omega;
}
				
