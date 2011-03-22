/* Count the number of lines in a text file */
unsigned int nb_lines( char* fn );
/* Create a gsl matrix from a text file */
gsl_matrix* file2matrix( char* fn, unsigned int col );
/* Return an int chosen randomly in [min:max] */
int random_int( int min, int max );
/* Return 0 w.p. 0.9 and 1 w.p. 0.1 */
int rand_1_in_10();
/* ||m1-m2||_2 */
double diff_norm( gsl_matrix* m1, gsl_matrix* m2 );
