#define _POSIX_C_SOURCE 1
#include <gsl/gsl_matrix.h>
#include <math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>

#define TRANS_WIDTH 7

/* Count the number of lines in a text file */
unsigned int nb_lines( char* fn ){
  unsigned int answer = 0;
  char str[1024];
  FILE* f = fopen( fn, "r" );
  while( fscanf( f, "%[^\n]", str ) != EOF ){
    answer++;
    fscanf( f, "\n" );
  }
  fclose( f );
  return answer;
}

/* Create a gsl matrix from a text file */
gsl_matrix* file2matrix( char* fn, unsigned int col ){
  unsigned int l = nb_lines( fn );
  gsl_matrix* answer = gsl_matrix_alloc( l, col );
  FILE* f = fopen( fn, "r" );
  gsl_matrix_fscanf( f, answer );
  fclose( f );
  return answer;
}


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
      double d_i = (double)i * M_PI/4.;
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
      double d_i = (double)i * M_PI/4.;
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
double g_dLambda_lstdmu = 0.1;
double g_dGamma_anirl = 0.9;//FIXME: Ill named. Should be something like g_dGamma_lstdmu
//I suspect it is used also in the monte carlo, hence the name when ANIRL was the only
//implemented IRL algorithm
gsl_matrix* (*g_fPsi)(gsl_matrix*) = psi;
gsl_matrix* (*g_fPhi)(gsl_matrix*) = phi;

gsl_matrix* lstd_mu_op_omega(  gsl_matrix* D_mu ){

  gsl_matrix* A = gsl_matrix_calloc( g_iK, g_iK );

  gsl_matrix* b = gsl_matrix_calloc( g_iK, g_iP );

  for( unsigned int i=0; i < D_mu->size1 - 1; i++ ){ //The last sample is unusable because we don't know pi(s')

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
    gsl_matrix_view adash_src = 
      gsl_matrix_submatrix( D_mu, i+1, g_iS, 1, g_iA );
    gsl_matrix_memcpy( &adash_dst.matrix, &adash_src.matrix );
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

       gsl_matrix_view psi_s = 
	 gsl_matrix_submatrix( D_mu, i, g_iS+g_iA+g_iS, 1, g_iP );
       gsl_matrix* delta_b = gsl_matrix_alloc( g_iK, g_iP );
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
  return omega_pi;
}




int main( int argc, char** argv ){
  gsl_matrix* D = file2matrix( argv[1], TRANS_WIDTH );

  gsl_matrix* D_mu = gsl_matrix_alloc( D->size1, g_iS + g_iA + g_iS + g_iP + 1 );

  for( int i = 0; i < D->size1 ; i++ ){
    gsl_matrix_view vsasdash_src = gsl_matrix_submatrix( D, i, 0, 1, g_iS + g_iA + g_iS );
    gsl_matrix_view vsasdash_dst = gsl_matrix_submatrix( D_mu, i, 0,
							 1, g_iS + g_iA + g_iS );
    gsl_matrix_memcpy( &(vsasdash_dst.matrix), &(vsasdash_src.matrix) );
    
    gsl_matrix_view vs = gsl_matrix_submatrix( D, i, 0, 1, g_iS );
    gsl_matrix* psi_s = g_fPsi( &(vs.matrix) );
    gsl_matrix_view vpsi_s = gsl_matrix_submatrix( D_mu, i, g_iS + g_iA + g_iS, 1, g_iP );
    gsl_matrix_transpose_memcpy( &(vpsi_s.matrix), psi_s );
    gsl_matrix_free( psi_s );

    gsl_matrix_view eoe_src = gsl_matrix_submatrix( D, i, g_iS + g_iA + g_iS + 1,
						    1, 1 );
    gsl_matrix_view eoe_dst = gsl_matrix_submatrix( D_mu, i, g_iS + g_iA + g_iS + g_iP,
						    1, 1 );
    gsl_matrix_memcpy( &(eoe_dst.matrix), &(eoe_src.matrix) );    
  }
  
  gsl_matrix* omega_pi = lstd_mu_op_omega( D_mu );
  
  for( int i = 0; i < omega_pi->size1; i++ ){
    for( int j = 0; j < omega_pi->size2; j++ ){
      printf("%e ",gsl_matrix_get( omega_pi, i, j ) );
    }
    printf("\n");
  }

  return 0;
}
