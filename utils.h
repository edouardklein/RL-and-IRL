/* Count the number of lines in a text file */
unsigned int nb_lines( char* fn );
/* Create a gsl matrix from a text file */
gsl_matrix* file2matrix( char* fn, unsigned int col );
/* Return an int chosen randomly in [min:max] */
int random_int( int min, int max );
