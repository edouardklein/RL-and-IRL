
extern double g_dEpsilon_anirl;

extern unsigned int g_iIt_max_anirl;

extern double g_dGamma_anirl;
extern double g_dGamma_lafem;

extern unsigned int g_iP;

extern gsl_matrix* (*g_fPsi)(gsl_matrix*);

extern gsl_matrix* g_mOmega_E;

extern gsl_matrix* (*g_fSimulator)(int);

extern unsigned int g_iNb_samples;

extern double g_dLambda_lstdmu; 

extern double g_dBest_error;
extern double g_dBest_true_error;
extern double g_dBest_diff;
extern double g_dBest_t;
extern gsl_matrix* g_mBest_omega;

extern gsl_matrix* (*g_fS_0)( void );
