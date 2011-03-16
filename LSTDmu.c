#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <math.h>
#include "greedy.h"
#include "LSPI.h"
#include "utils.h"
#include "GridWorld_simulator.h"

/**/
gsl_matrix* monte_carlo_mu( gsl_matrix* states,gsl_matrix* EOEs,
			    double gamma,
			    gsl_matrix* (*psi)(gsl_matrix*));
/**/

#define LAMBDA 0.1 /* Regularisation is needed when computing*/
/* the mu of a good policy because only a handful of state action
 couples are visited, thus making the A matrix singular */

/* Compute an estimate of \mu(s_0) using the LSTD\mu algorithm,
   in its on policy form. Typically used for the expert
*/
gsl_matrix* lstd_mu_op(  gsl_matrix* D_mu, unsigned int k,
			 unsigned int p,
			 unsigned int s, unsigned int a,
			 gsl_matrix* (*phi)(gsl_matrix*),
			 gsl_matrix* (*psi)(gsl_matrix*),
			 double gamma ){
  gsl_matrix* s_0 = gsl_matrix_alloc( 1, s );
  gsl_matrix* a_0 = gsl_matrix_alloc( 1, a );
  gsl_matrix_view s_0_src = gsl_matrix_submatrix( D_mu, 0, 0,
						  1, s );
  gsl_matrix_view a_0_src = gsl_matrix_submatrix( D_mu, 0, s,
						  1, a );
  gsl_matrix_memcpy( s_0, &s_0_src.matrix );
  gsl_matrix_memcpy( a_0, &a_0_src.matrix );
  // \tilde A \leftarrow 0
  gsl_matrix* A = gsl_matrix_calloc( k, k );
  // \tilde b \leftarrow 0
  gsl_matrix* b = gsl_matrix_calloc( k, p );
  // for each (s,a,s',\psi(s),eoe) \in D_mu
  /**/
  //printf("Nb of samples : %d\n",D_mu->size1);
  /**/
  for( unsigned int i=0; i < D_mu->size1 ; i++ ){
    // \tilde A \leftarrow \tilde A + \phi(s,a)\left(\phi(s,a) 
    // - \gamma \phi(s',a')\right)^T
    gsl_matrix_view sa = gsl_matrix_submatrix(D_mu,i,0, 1, s+a);
    gsl_matrix* phi_sa = phi( &sa.matrix );
    gsl_matrix* sa_dash = gsl_matrix_calloc( 1, s+a );
    gsl_matrix_view sdash_dst = gsl_matrix_submatrix( sa_dash, 
						      0, 0,
						      1, s);
    gsl_matrix_view sdash_src = gsl_matrix_submatrix( D_mu, 
						      i, s+a, 
						      1, s);
    gsl_matrix_memcpy( &sdash_dst.matrix, &sdash_src.matrix );
    gsl_matrix_view adash_dst = gsl_matrix_submatrix( sa_dash,
						      0, s, 
						      1, a);
    int a_index = i+1;
    if( a_index >= D_mu->size1 )
      a_index = i;
    gsl_matrix_view adash_src = gsl_matrix_submatrix( D_mu, 
						      a_index,s,
						      1, a);
    gsl_matrix_memcpy( &adash_dst.matrix, &adash_src.matrix );
    /**/
    /* printf("(s,a,s',a') = (%d,%d),%d,(%d,%d),%d\n", */
    /* 	   (int) gsl_matrix_get( &sa.matrix, 0, 0 ), */
    /* 	   (int)gsl_matrix_get( &sa.matrix, 0, 1 ), */
    /* 	   (int)gsl_matrix_get( &sa.matrix, 0, 2 ), */
    /* 	   (int)gsl_matrix_get( sa_dash, 0, 0 ), */
    /* 	   (int)gsl_matrix_get( sa_dash, 0, 1 ), */
    /* 	   (int)gsl_matrix_get( sa_dash, 0, 2 ) */
    /* 	   ); */
    /**/
    gsl_matrix* phi_dash = phi( sa_dash );
    gsl_matrix_scale( phi_dash, gamma );
    double eoe = gsl_matrix_get( D_mu, i, s+a+s+p );
    gsl_matrix_scale( phi_dash, eoe );
    gsl_matrix* delta_phi = gsl_matrix_calloc( k, 1 );
    gsl_matrix_memcpy( delta_phi, phi_sa );
    gsl_matrix_sub( delta_phi, phi_dash );
    gsl_matrix* deltaA = gsl_matrix_calloc( k, k );
    gsl_blas_dgemm( CblasNoTrans, CblasTrans, 1., 
		      phi_sa, delta_phi, 0., deltaA);
    gsl_matrix_add( A, deltaA );
    //\tilde b \leftarrow \tilde b + \phi(s,a)\psi(s)^T
    gsl_matrix_view psi_s = 
      gsl_matrix_submatrix( D_mu, i, s+a+s, 1, p );
    gsl_matrix* delta_b = gsl_matrix_alloc( k, p );
    /* \psi(s) is in line in the code but in column in the 
     comments */
    gsl_blas_dgemm( CblasNoTrans, CblasNoTrans, 1.0,
		    phi_sa, &psi_s.matrix, 0.0, delta_b );
    gsl_matrix_add( b, delta_b );
    gsl_matrix_free( deltaA );
    gsl_matrix_free( delta_phi );
    gsl_matrix_free( phi_dash );
    gsl_matrix_free( sa_dash );
    gsl_matrix_free( phi_sa );
    gsl_matrix_free( delta_b );
  }
  /*\tilde \omega^\pi \leftarrow 
    (\tildeA + \lambda Id) ^{-1}\tilde b */
  /* for( int j = 0; j<100;j++){ */
  /*   for(int k=0; k<100;k++){ */
  /*     printf("%1lf ",gsl_matrix_get(A,j,k)); */
  /*   } */
  /*   printf("\n"); */
  /* } */
  gsl_matrix* lambdaI = gsl_matrix_alloc( A->size1, A->size2 );
  gsl_matrix_set_identity( lambdaI );
  gsl_matrix_scale( lambdaI, LAMBDA );
  gsl_matrix_add( A, lambdaI );
  gsl_matrix_free( lambdaI );
  gsl_matrix* omega_pi = gsl_matrix_alloc( k, p );
  gsl_permutation* _p = gsl_permutation_alloc( k );
  int signum;
  gsl_linalg_LU_decomp( A, _p, &signum );
  for( unsigned int i = 0 ; i < p ; i++ ){
    gsl_vector_view b_v = gsl_matrix_column( b, i );
    gsl_vector_view o_v = gsl_matrix_column( omega_pi, i );
    gsl_linalg_LU_solve( A, _p, &b_v.vector, &o_v.vector );
  }
  gsl_permutation_free( _p );
  gsl_matrix_free( A );
  gsl_matrix_free( b );
   //\mu_\pi(s) \leftarrow \tilde\omega_\pi^T\phi(s,\pi(s))
  gsl_matrix* s_pi_s = gsl_matrix_alloc( 1, s+a );
  gsl_matrix_view s_dst = gsl_matrix_submatrix( s_pi_s, 
						0, 0,
						1, s);
  gsl_matrix_memcpy( &s_dst.matrix, s_0 );
  gsl_matrix_view pi_s_dst = gsl_matrix_submatrix( s_pi_s,
						   0, s, 
						   1, a);
  gsl_matrix_memcpy( &pi_s_dst.matrix, a_0 );
  gsl_matrix* phi_s_pi_s = phi( s_pi_s );
  gsl_matrix* mu = gsl_matrix_alloc( p, 1 );
  gsl_blas_dgemm( CblasTrans, CblasNoTrans, 1.0,
		  omega_pi, phi_s_pi_s, 0.0, mu );
  gsl_matrix_free( omega_pi );
  gsl_matrix_free( s_pi_s );
  gsl_matrix_free( a_0 );
  gsl_matrix_free( s_0 );
  return mu;
}

/* Compute an estimate of \mu(s_0) using the LSTD\mu algorithm
*/
gsl_matrix* lstd_mu(  gsl_matrix* D_mu, unsigned int k,
		      unsigned int p,
		      unsigned int s, unsigned int a,
		      gsl_matrix* (*phi)(gsl_matrix*),
		      gsl_matrix* (*psi)(gsl_matrix*),
		      double gamma,
		      gsl_matrix* (*pi)(gsl_matrix*), 
		      gsl_matrix* s_0){
  // \tilde A \leftarrow 0
  gsl_matrix* A = gsl_matrix_calloc( k, k );
  // \tilde b \leftarrow 0
  gsl_matrix* b = gsl_matrix_calloc( k, p );
  // for each (s,a,s',\psi(s),eoe) \in D_mu
   /**/
  //printf("Nb of samples : %d\n",D_mu->size1);
  /**/
  for( unsigned int i=0; i < D_mu->size1 ; i++ ){
    // \tilde A \leftarrow \tilde A + \phi(s,a)\left(\phi(s,a) 
    // - \gamma \phi(s',\pi(s'))\right)^T
    gsl_matrix_view sa = gsl_matrix_submatrix(D_mu,i,0, 1, s+a);
    gsl_matrix* phi_sa = phi( &sa.matrix );
    gsl_matrix* sa_dash = gsl_matrix_calloc( 1, s+a );
    gsl_matrix_view sdash_dst = gsl_matrix_submatrix( sa_dash, 
						      0, 0,
						      1, s);
    gsl_matrix_view sdash_src = gsl_matrix_submatrix( D_mu, 
						      i, s+a, 
						      1, s);
    gsl_matrix_memcpy( &sdash_dst.matrix, &sdash_src.matrix );
    gsl_matrix_view adash_dst = gsl_matrix_submatrix( sa_dash,
						      0, s, 
						      1, a);
    gsl_matrix* adash_src = pi( &sdash_src.matrix );
    gsl_matrix_memcpy( &adash_dst.matrix, adash_src );
    /**/
    /* printf("(s,a,s',a') = (%d,%d),%d,(%d,%d),%d\n", */
    /* 	   (int) gsl_matrix_get( &sa.matrix, 0, 0 ), */
    /* 	   (int)gsl_matrix_get( &sa.matrix, 0, 1 ), */
    /* 	   (int)gsl_matrix_get( &sa.matrix, 0, 2 ), */
    /* 	   (int)gsl_matrix_get( sa_dash, 0, 0 ), */
    /* 	   (int)gsl_matrix_get( sa_dash, 0, 1 ), */
    /* 	   (int)gsl_matrix_get( sa_dash, 0, 2 ) */
    /* 	   ); */
    /**/
    gsl_matrix* phi_dash = phi( sa_dash );
    gsl_matrix_scale( phi_dash, gamma );
    double eoe = gsl_matrix_get( D_mu, i, s+a+s+p );
    gsl_matrix_scale( phi_dash, eoe );
    gsl_matrix* delta_phi = gsl_matrix_calloc( k, 1 );
    gsl_matrix_memcpy( delta_phi, phi_sa );
    gsl_matrix_sub( delta_phi, phi_dash );
    gsl_matrix* deltaA = gsl_matrix_calloc( k, k );
    gsl_blas_dgemm( CblasNoTrans, CblasTrans, 1., 
		      phi_sa, delta_phi, 0., deltaA);
    gsl_matrix_add( A, deltaA );
    /**/
    /* printf("EOE : %lf\n",eoe); */
    /* printf("MATRICE delta A\n"); */
    /* for( int j = 0; j<16;j++){ */
    /*   for(int k=0; k<16;k++){ */
    /* 	printf("% 6.1g ",gsl_matrix_get(deltaA,j,k)); */
    /*   } */
    /*   printf("\n"); */
    /* } */

    /* printf("phi(sa) : \n"); */
    /* for( int j=0;j<16;j++){ */
    /*   printf("%.2lf ",gsl_matrix_get( phi_sa, j, 0 )); */
    /* } */
    /* printf("\n"); */
    /* printf("phi(s'pi(s')) : \n"); */
    /* for( int j=0;j<16;j++){ */
    /*   printf("%.2lf ",gsl_matrix_get( phi_dash, j, 0 )); */
    /* } */
    /* printf("\n"); */
    /* printf("delta phi : \n"); */
    /* for( int j=0;j<16;j++){ */
    /*   printf("%.2lf ",gsl_matrix_get( delta_phi, j, 0 )); */
    /* } */
    /* printf("\n"); */
    /* printf("Matrice delta A \n"); */
    /* for( int j=0; j<16;j++){ */
    /*   for( int k=0; k<16;k++){ */
    /* 	printf("%.2lf ",gsl_matrix_get(deltaA,j,k)); */
    /*   } */
    /*   printf("\n"); */
    /* } */
    /* printf("--------------\n"); */
    /**/
    //\tilde b \leftarrow \tilde b + \phi(s,a)\psi(s)^T
    gsl_matrix_view psi_s = 
      gsl_matrix_submatrix( D_mu, i, s+a+s, 1, p );
    gsl_matrix* delta_b = gsl_matrix_alloc( k, p );
    /*\psi(s) is in line in the code but in column 
      in the comments*/ 
    gsl_blas_dgemm( CblasNoTrans, CblasNoTrans, 1.0,
		    phi_sa, &psi_s.matrix, 0.0, delta_b );
    gsl_matrix_add( b, delta_b );
    /**/
    /* printf("MATRICE delta b\n"); */
    /* for( int j = 0; j<16;j++){ */
    /*   for(int k=0; k<4;k++){ */
    /* 	printf("% 6.1g ",gsl_matrix_get(delta_b,j,k)); */
    /*   } */
    /*   printf("\n"); */
    /* } */
    /**/
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
  gsl_matrix_scale( lambdaI, LAMBDA );
  gsl_matrix_add( A, lambdaI );
  gsl_matrix_free( lambdaI );
  gsl_matrix* omega_pi = gsl_matrix_alloc( k, p );
  gsl_matrix* A_back = gsl_matrix_alloc( A->size1, A->size2 );
  gsl_matrix_memcpy( A_back, A );
  gsl_permutation* _p = gsl_permutation_alloc( k );
  int signum;
  gsl_linalg_LU_decomp( A, _p, &signum );
  for( unsigned int i = 0 ; i < p ; i++ ){
    gsl_vector_view b_v = gsl_matrix_column( b, i );
    gsl_vector_view o_v = gsl_matrix_column( omega_pi, i );
    gsl_linalg_LU_solve( A, _p, &b_v.vector, &o_v.vector );
  }
  gsl_permutation_free( _p );
  /**/
  /* printf("MATRICE A\n"); */
  /* for( int j = 0; j<16;j++){ */
  /*   for(int k=0; k<16;k++){ */
  /*     printf("% 6.5g ",gsl_matrix_get(A_back,j,k)); */
  /*   } */
  /*   printf("\n"); */
  /* } */
  /* printf("MATRICE omega\n"); */
  /* for( int j = 0; j<16;j++){ */
  /*   for(int k=0; k<4;k++){ */
  /*     printf("% 6.5g ",gsl_matrix_get(omega_pi,j,k)); */
  /*   } */
  /*   printf("\n"); */
  /* } */
  /* gsl_matrix* blop = gsl_matrix_alloc( b->size1, b->size2 ); */
  /* gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0, */
  /* 		 A_back,omega_pi,0.0,blop); */
  /* printf("MATRICE b\n"); */
  /* for( int j = 0; j<16;j++){ */
  /*   for(int k=0; k<4;k++){ */
  /*     printf("% 6.5g ",gsl_matrix_get(b,j,k)); */
  /*   } */
  /*   printf("\n"); */
  /* } */
  /* printf("MATRICE tilde b\n"); */
  /* for( int j = 0; j<16;j++){ */
  /*   for(int k=0; k<4;k++){ */
  /*     printf("% 6.5g ",gsl_matrix_get(blop,j,k)); */
  /*   } */
  /*   printf("\n"); */
  /* } */
  /* gsl_matrix_free( blop ); */
  /**/
  gsl_matrix_free( A );
  gsl_matrix_free( b );
  //\mu_\pi(s) \leftarrow \tilde\omega_\pi^T\phi(s,\pi(s))
  gsl_matrix* s_pi_s = gsl_matrix_alloc( 1, s+a );
  gsl_matrix_view s_dst = gsl_matrix_submatrix( s_pi_s, 
						0, 0,
						1, s);
  gsl_matrix_memcpy( &s_dst.matrix, s_0 );
  gsl_matrix_view pi_s_dst = gsl_matrix_submatrix( s_pi_s,
						    0, s, 
						    1, a);
  gsl_matrix* pi_s_src = pi( s_0 );
  gsl_matrix_memcpy( &pi_s_dst.matrix, pi_s_src );
  gsl_matrix* phi_s_pi_s = phi( s_pi_s );
  gsl_matrix* mu = gsl_matrix_alloc( p, 1 );
  gsl_blas_dgemm( CblasTrans, CblasNoTrans, 1.0,
		  omega_pi, phi_s_pi_s, 0.0, mu );
  gsl_matrix_free( omega_pi );
  gsl_matrix_free( s_pi_s );
  gsl_matrix_free( pi_s_src );
  return mu;
}

/* Abbeel and Ng's IRL algorithm (ANIRL), with the projection 
   method, LSTDMu estimation and LSPI as the MDP solver.
   Given an expert's trace, returns the \omega 
   matrix which defines the optimal policy as found by LSPI
   under the reward R = \theta^T\psi. 
   Note that \omega \equiv \pi
*/
gsl_matrix* 
proj_lstd_lspi_ANIRL( gsl_matrix* expert_trans,
		      gsl_matrix* D,
		      unsigned int s, unsigned int a,
		      unsigned int k, unsigned int p,
		      double gamma, double gamma_lspi,
		      double epsilon, double epsilon_lspi,
		      gsl_matrix* (*phi)(gsl_matrix*),
		      gsl_matrix* (*psi)(gsl_matrix*)){
  gsl_matrix* s_0 = gsl_matrix_alloc( 1, s );
  gsl_matrix_view s_0_src = gsl_matrix_submatrix( expert_trans,
						  0, 0,
						  1, s );
  gsl_matrix_memcpy( s_0, &s_0_src.matrix );
  unsigned int m = expert_trans->size1 + D->size1;
  gsl_matrix* omega_0 = gsl_matrix_calloc( k, 1 );
  /* \omega \leftarrow 0 */
  gsl_matrix* omega = gsl_matrix_calloc( k, 1 );
  /* D_\mu \leftarrow D 
     D_\mu.r \leftarrow \psi(D.s) */
  gsl_matrix* D_mu = gsl_matrix_alloc( D->size1, s+a+s+p+1 );
  gsl_matrix_view Dsas = gsl_matrix_submatrix( D, 0, 0, 
					       D->size1, s+a+s);
  gsl_matrix_view Dmusas = gsl_matrix_submatrix( D_mu, 0, 0, 
					       D->size1, s+a+s);
  gsl_matrix_memcpy( &Dmusas.matrix, &Dsas.matrix );
  for( unsigned int i = 0 ; i<D->size1 ; i++ ){
    gsl_matrix_view vS = gsl_matrix_submatrix( D, i, 0, 1, s );
    gsl_matrix* psi_s_src = psi( &vS.matrix );
    gsl_matrix_view psi_s_dst = 
      gsl_matrix_submatrix( D_mu, i, s+a+s, 1, p );
    /* D.\psi(s) is in line in the code but in columns in
       the comments */
    gsl_matrix_transpose_memcpy( &psi_s_dst.matrix, psi_s_src );
    gsl_matrix_free( psi_s_src );
  }
  gsl_matrix_view Deoe = gsl_matrix_submatrix( D, 0, s+a+s+1,
					       D->size1, 1 );
  gsl_matrix_view Dmueoe = gsl_matrix_submatrix( D_mu,
						 0, s+a+s+p,
						 D->size1, 1 ); 
  gsl_matrix_memcpy( &Dmueoe.matrix, &Deoe.matrix );
  /* \mu \leftarrow LSTD_\mu( D_\mu, k, p, s, a, \phi,
     \psi, \gamma, \pi ) */
  g_mOmega = gsl_matrix_calloc( k, 1 );
  gsl_matrix_memcpy( g_mOmega, omega );
  g_mActions = file2matrix( ACTION_FILE, g_iA );
  gsl_matrix* mu = 
    lstd_mu( D_mu, k, p, s, a, phi, psi, gamma, 
	     &greedy_policy, s_0 );
  gsl_matrix_free( g_mOmega );
  gsl_matrix_free( g_mActions );
  /* D_E.r \leftarrow \psi(s) */
  gsl_matrix* D_E = gsl_matrix_alloc( expert_trans->size1, 
				      s+a+s+p+1 );
  gsl_matrix_view DEsas_dst = gsl_matrix_submatrix( D_E, 0, 0, 
					      D_E->size1,s+a+s);
  gsl_matrix_view DEsas_src = gsl_matrix_submatrix(expert_trans,
						   0, 0, 
					     D_E->size1, s+a+s);
  gsl_matrix_memcpy( &DEsas_dst.matrix, &DEsas_src.matrix );
  for( unsigned int i = 0 ; i<D_E->size1 ; i++ ){
    gsl_matrix_view vS = gsl_matrix_submatrix(D_E, i, 0, 1, s );
    gsl_matrix* psi_s_src = psi( &vS.matrix );
    gsl_matrix_view psi_s_dst = 
      gsl_matrix_submatrix( D_E, i, s+a+s, 1, p );
    /* D.\psi(s) is in line in the code but in columns in
       the comments */
    gsl_matrix_transpose_memcpy( &psi_s_dst.matrix, psi_s_src );
    gsl_matrix_free( psi_s_src );
  }
  gsl_matrix_view DEeoe_dst = gsl_matrix_submatrix( D_E, 
						    0, s+a+s+p,
						D_E->size1, 1 );
  gsl_matrix_view DEeoe_src = gsl_matrix_submatrix(expert_trans,
						 0, s+a+s+1,
						 D_E->size1,1); 
  gsl_matrix_memcpy( &DEeoe_dst.matrix, &DEeoe_src.matrix );
  /* \mu_E \leftarrow on-LSTD_\mu( D_E, k, p, s, a, \psi,
     \phi, \gamma) */
  gsl_matrix* mu_E = 
    lstd_mu_op( D_E, k, p, s, a, phi, psi, gamma );
  /* \theta \leftarrow {\mu_E - \mu\over ||\mu_E - \mu||_2} */
  gsl_matrix* theta = gsl_matrix_alloc( mu->size1, mu->size2 );
  gsl_matrix_memcpy( theta, mu_E );
  gsl_matrix_sub( theta, mu );
  gsl_vector_view theta_v = gsl_matrix_column( theta, 0 );
  double theta_norm = gsl_blas_dnrm2( &theta_v.vector );
  if( theta_norm != 0 )
    gsl_matrix_scale( theta, 1./theta_norm );
  /* \bar\mu \leftarrow \mu*/
  gsl_matrix* bar_mu = gsl_matrix_alloc( mu->size1, mu->size2 );
  gsl_matrix_memcpy( bar_mu, mu );
  /* t = ||\mu_E - \bar\mu||_2*/
  gsl_vector_view barmu_v = gsl_matrix_column( bar_mu, 0 );
  gsl_vector_view muE_v = gsl_matrix_column( mu_E, 0 );
  gsl_vector* diff = gsl_vector_calloc( mu->size1 );
  gsl_vector_memcpy( diff, &barmu_v.vector );
  gsl_vector_sub( diff, &muE_v.vector );
  double t = gsl_blas_dnrm2( diff );
  unsigned int nb_it = 0;
    /**/
    /* /\* \mu \leftarrow mc( D_\pi, \gamma, \psi ) *\/ */
    /* gsl_matrix_view states =  */
    /*   gsl_matrix_submatrix( expert_trans, 0, 0,  */
    /* 			    expert_trans->size1, s ); */
    /* gsl_matrix_view EOEs =  */
    /*   gsl_matrix_submatrix( expert_trans, 0, 2*s+a+1,  */
    /* 			    expert_trans->size1, 1 ); */
    /* gsl_matrix* muE_mc =  */
    /*   monte_carlo_mu( &states.matrix, &EOEs.matrix,gamma, psi ); */
    /* printf("Mu E : (by MC)\n"); */
    /* for( int i = 0; i<mu->size1;i++){ */
    /*   printf("%lf ",gsl_matrix_get(muE_mc,i,0)); */
    /* } */
    /* printf("\n"); */
    /* gsl_matrix_free(muE_mc); */
    /* printf("Mu E : (by LSTD) \n"); */
    /* for( int i = 0; i<mu->size1;i++){ */
    /*   printf("%lf ",gsl_matrix_get(mu_E,i,0)); */
    /* } */
    /* printf("\n"); */
    //exit(-1);
    /**/
  while( t > epsilon ){
    printf("%d %d %d %lf\n", m, nb_it, g_iNb_samples, t );
    /**/
    /* D_\pi \leftarrow simulator( m, \omega ) */
    /* g_mOmega = gsl_matrix_calloc( k, 1 ); */
    /* gsl_matrix_memcpy( g_mOmega, omega ); */
    /* g_mActions = file2matrix( ACTION_FILE, g_iA ); */
    /* gsl_matrix* trans = gridworld_simulator( 30 ); */
    /* gsl_matrix_free( g_mOmega ); */
    /* gsl_matrix_free( g_mActions ); */
    /* \mu \leftarrow mc( D_\pi, \gamma, \psi ) */
    /* gsl_matrix_view states =  */
    /*   gsl_matrix_submatrix( trans, 0, 0, trans->size1, s ); */
    /* gsl_matrix_view EOEs =  */
    /*   gsl_matrix_submatrix( trans, 0, 2*s+a+1, trans->size1, 1 ); */
    /* gsl_matrix* mu_mc =  */
    /*   monte_carlo_mu( &states.matrix, &EOEs.matrix,gamma, psi ); */
    /* gsl_matrix_free( trans ); */
    /* printf("Mu : (by MC)\n"); */
    /* for( int i = 0; i<mu->size1;i++){ */
    /*   printf("%lf ",gsl_matrix_get(mu_mc,i,0)); */
    /* } */
    /* printf("\n"); */
    /* gsl_matrix_free(mu_mc); */
    /* printf("Mu : (by LSTD) \n"); */
    /* for( int i = 0; i<mu->size1;i++){ */
    /*   printf("%lf ",gsl_matrix_get(mu,i,0)); */
    /* } */
    /* printf("\n"); */
    //exit(1);
    /**/
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
    /* \mu \leftarrow LSTD_\mu( D_\mu, k, p, s, a, \phi,
       \psi, \gamma, \pi ) */
    g_mOmega = gsl_matrix_calloc( k, 1 );
    gsl_matrix_memcpy( g_mOmega, omega );
    g_mActions = file2matrix( ACTION_FILE, g_iA );
    gsl_matrix_free( mu );
    mu = lstd_mu( D_mu, k, p, s, a, phi, psi, gamma, 
		  &greedy_policy, s_0);
    gsl_matrix_free( g_mOmega );
    gsl_matrix_free( g_mActions );
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
    /* \theta \leftarrow \mu_E - \bar\mu */
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
  printf("%d %d %d %lf\n", m, nb_it, g_iNb_samples, t );
  gsl_matrix_free( omega_0 );
  gsl_matrix_free( mu );
  gsl_matrix_free( mu_E );
  gsl_matrix_free( bar_mu );
  gsl_vector_free( diff );
  gsl_matrix_free( theta );
  return omega;
}
				
