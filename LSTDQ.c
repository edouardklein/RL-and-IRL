#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>

gsl_matrix* lstd_q( gsl_matrix* D, unsigned int k,
		    unsigned int s, unsigned int a,
		    gsl_matrix* (*phi)(gsl_matrix*),
		    double gamma,
		    gsl_matrix* (*pi)(gsl_matrix*) ){
  // \tilde A \leftarrow 0
  gsl_matrix* A = gsl_matrix_calloc( k, k );
  // \tilde b \leftarrow 0
  gsl_matrix* b = gsl_matrix_calloc( k, 1 );
  // for each (s,a,s',r,eoe) \in D
  for( unsigned int i=0; i < D->size1 ; i++ ){
    // \tilde A \leftarrow \tilde A + \phi(s,a)\left(\phi(s,a) 
    // - \gamma \phi(s',\pi(s'))\right)^T
    gsl_matrix_view sa = gsl_matrix_submatrix(D, i, 0, 1, s+a);
    gsl_matrix* phi_sa = phi( &sa.matrix );
    gsl_matrix* sa_dash = gsl_matrix_alloc( 1, s+a );
    gsl_matrix_view sdash_dst = gsl_matrix_submatrix( sa_dash, 
						      0, 0,
						      1, s);
    gsl_matrix_view sdash_src = gsl_matrix_submatrix( D, 
						      i, s+a, 
						      1, s);
    gsl_matrix_memcpy( &sdash_dst.matrix, &sdash_src.matrix );
    gsl_matrix_view adash_dst = gsl_matrix_submatrix( sa_dash,
						      0, s, 
						      1, a);
    gsl_matrix* adash_src = pi( &sdash_src.matrix );
    gsl_matrix_memcpy( &adash_dst.matrix, adash_src );
    gsl_matrix* phi_dash = phi( sa_dash );
    gsl_matrix_scale( phi_dash, gamma );
    double eoe = gsl_matrix_get( D, i, s+a+s+1 );
    gsl_matrix_scale( phi_dash, eoe );
    gsl_matrix* delta_phi = gsl_matrix_calloc( k, 1 );
    gsl_matrix_memcpy( delta_phi, phi_sa );
    gsl_matrix_sub( delta_phi, phi_dash );
    gsl_matrix* deltaA = gsl_matrix_calloc( k, k );
    gsl_blas_dgemm( CblasNoTrans, CblasTrans, 1., 
		      phi_sa, delta_phi, 0., deltaA);
    gsl_matrix_add( A, deltaA );
    //\tilde b \leftarrow \tilde b + \phi(s,a)r
    double r = gsl_matrix_get( D, i, s+a+s );
    gsl_matrix_scale( phi_sa, r ); //phi_sa is now delta_b
    gsl_matrix_add( b, phi_sa );
    gsl_matrix_free( deltaA );
    gsl_matrix_free( delta_phi );
    gsl_matrix_free( phi_dash );
    gsl_matrix_free( adash_src );
    gsl_matrix_free( sa_dash );
    gsl_matrix_free( phi_sa );
  }
  //\tilde \omega^\pi \leftarrow \tilde A^{-1}\tilde b
  gsl_vector_view b_v = gsl_matrix_column( b, 0 );
  gsl_matrix* omega_pi = gsl_matrix_alloc( k, 1 );
  gsl_vector_view o_v = gsl_matrix_column( omega_pi, 0 );
  gsl_permutation* p = gsl_permutation_alloc( k );
  int signum;
  gsl_linalg_LU_decomp( A, p, &signum );
  gsl_linalg_LU_solve( A, p, &b_v.vector, &o_v.vector );
  gsl_matrix_free( A );
  gsl_matrix_free( b );
  gsl_permutation_free( p );
  return omega_pi;
}
