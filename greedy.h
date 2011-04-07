/* Return the action a for which Q(state,a) is max */
gsl_matrix* greedy_policy( gsl_matrix* state );

/* Return V(s) found by maximizing Q(s,a) over a */
double greedy_value_function( gsl_matrix* state );
