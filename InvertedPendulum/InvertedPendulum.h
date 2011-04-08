#define LEFT 0
#define NONE 1
#define RIGHT 2
#define PI (3.1415926536)
void iv_init( double* pos, double* speed );
void iv_step( double state_p, double state_v, 
	      unsigned int action,
	      double* next_state_p, double* next_state_v, 
	      double* reward, int* eoe );
