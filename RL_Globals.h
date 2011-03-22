/* Some global variables so that argument list of the 
   different functions are less than 8 lines long.
   They are just declared here. Could be a good
   idea to set all of them once and for all in the same
   place
*/

/*Value that stops LSPI*/
extern double g_dEpsilon_lspi;

/* Number of iterations after which the LSPI loop
 is stopped, even if the norm < g_dEpsilon_lspi criterion is 
 not matched*/
extern unsigned int g_iIt_max_lspi;

//Dimension of the feature space
extern unsigned int g_iK;

/*S\times A\rightarrow \mathbb{R}^k features over which Q is 
 approximated */
extern gsl_matrix* (*g_fPhi)(gsl_matrix*);

//Dimension of the action space
extern unsigned int g_iA; 

//Dimension of the state space
extern unsigned int g_iS;

//Discount factor for LSPI and LSTDQ
extern double g_dGamma_lstdq;

//Omega matrix used by greedy_policy
extern gsl_matrix* g_mOmega; //Omega

//All actions, one per line, used by greedy_policy
extern gsl_matrix* g_mActions; 

/* Regularisation is needed when computing Q and D does not
   cover the whole S\times A space, thus making the A matrix 
   singular */
extern double g_dLambda_lstdQ; 
