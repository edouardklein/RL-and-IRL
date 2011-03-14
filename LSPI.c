#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include "LSTDQ.h"
#include "utils.h"
#include "greedy.h"
#include "LSPI.h"

/* #include "GridWorld.h" */
/* void print_action( int x, int y ){ */
/*   gsl_matrix* state = gsl_matrix_alloc( 1, 2 ); */
/*   gsl_matrix_set( state, 0, 0, (double)x ); */
/*   gsl_matrix_set( state, 0, 1, (double)y ); */
/*   gsl_matrix* a = greedy_policy( state ); */
/*   int action = (int)gsl_matrix_get( a, 0, 0 ); */
/*   gsl_matrix_free( a ); */
/*   gsl_matrix_free( state ); */
/*   switch( action ){ */
/*   case UP: */
/*     printf("UP   "); */
/*     break; */
/*   case DOWN: */
/*     printf("DOWN "); */
/*     break; */
/*   case LEFT: */
/*     printf("LEFT "); */
/*     break; */
/*   case RIGHT: */
/*     printf("RIGHT"); */
/*     break; */
/*   } */
/* } */

/* void print_info(){ */
/*   for( int x = 1; x<=5 ; x++ ){ */
/*     for( int y = 1; y<=5 ; y++ ){ */
/*       printf("(%d,%d) : ",x,y); */
/*       print_action( x, y ); */
/*       printf(" "); */
/*     } */
/*     printf("\n"); */
/*   } */
/* } */


gsl_matrix* lspi( gsl_matrix* D, unsigned int k, 
		  unsigned int s, unsigned int a,
		  gsl_matrix* (*phi)(gsl_matrix*),
		  double gamma, double epsilon,
		  gsl_matrix* omega_0 ){
  g_iA = a;
  g_iS = s;
  g_iK = k;
  g_fPhi = phi;
  g_mActions = file2matrix( ACTION_FILE, g_iA );
  g_mOmega = gsl_matrix_alloc( omega_0->size1, 
			       omega_0->size2 );
  unsigned int nb_iterations = 0;
  //\omega'\leftarrow \omega_0
  gsl_matrix* omega_dash = gsl_matrix_alloc( omega_0->size1,
					     omega_0->size2 );
  gsl_matrix_memcpy( omega_dash, omega_0 );
  double norm;
  //FILE* f = fopen( "/dev/stdout", "w" );
  //Repeat
  do{
    //\omega \leftarrow \omega'
    gsl_matrix_memcpy( g_mOmega, omega_dash );
    //printf("Omega vaut : \n");
    //gsl_matrix_fprintf( f, omega_dash, "%f" );
    //printf("La politique associée : \n");
    //print_info( );
    //\omega' \leftarrow lstd_q(D,k,\phi,\gamma,\omega)
    gsl_matrix_free( omega_dash );
    omega_dash = lstd_q( D, k, s, a, phi, gamma, 
			 &greedy_policy );
    //printf("Après LSTDQ, omega' vaut : \n");
    //gsl_matrix_fprintf( f, omega_dash, "%f" );
    //until ( ||\omega'-\omega|| < \epsilon )
    gsl_vector_view o_v = gsl_matrix_column( g_mOmega, 0 );
    gsl_vector_view o_d_v = gsl_matrix_column( omega_dash, 0 );
    gsl_vector* diff = gsl_vector_calloc( k );
    gsl_vector_memcpy( diff, &o_v.vector );
    gsl_vector_sub( diff, &o_d_v.vector );
    norm = gsl_blas_dnrm2( diff );
    nb_iterations++;
    //printf("LSPI : Norme %lf\n",norm);
  }while( norm >= epsilon && nb_iterations < NB_ITERATIONS_MAX);
  gsl_matrix_free( g_mOmega );
  gsl_matrix_free( g_mActions );
  //We return omega' and not omega
  return omega_dash;
}
