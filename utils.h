
unsigned int nb_lines( char* fn );

gsl_matrix* file2matrix( char* fn, unsigned int col );

int random_int( int min, int max ); 

int rand_1_in_10();

double diff_norm( gsl_matrix* m1, gsl_matrix* m2 );
