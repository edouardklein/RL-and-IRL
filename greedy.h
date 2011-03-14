extern unsigned int g_iS; //Dimension of S
extern unsigned int g_iA; //Dimension on A
extern unsigned int g_iK; //Number of features
extern gsl_matrix* g_mOmega; //Omega
extern gsl_matrix* (*g_fPhi)(gsl_matrix*); //\phi
extern gsl_matrix* g_mActions; //All actions, one per line

/* Returns the action a for which Q(state,a) is max */
gsl_matrix* greedy_policy( gsl_matrix* state );
