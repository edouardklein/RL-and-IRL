#include <stdio.h>
#include <gsl/gsl_matrix.h>
#include <stdlib.h>

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

/* Return an int chosen randomly in [min:max] */
int random_int( int min, int max ){
  return rand()%(max-min+1) + min;
}

/* Return 0 w.p. 0.9 and 1 w.p. 0.1 */
int rand_1_in_10(){
  if( (double)rand() > (double)RAND_MAX*0.1 ){
    return 0;
  }
  return 1;
}
