#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_matrix.h>
#include "InvertedPendulum.h"
#include "utils.h"

void iv_init( double* pos, double* speed ){
  *pos = 0.04;
  //(double)rand()/(double)RAND_MAX*0.01;
  *speed = -0.02;
  //  (double)rand()/(double)RAND_MAX*0.01;
  int sign = random_int( 0, 1 );
  if (sign == 0){
    //*pos=-*pos;
  }
  sign = random_int( 0, 1 );
  if (sign == 0){
    //*speed=-*speed;
  }
}

void iv_step( double state_p, double state_v, 
	      unsigned int action,
	      double* next_state_p, double* next_state_v, 
	      double* reward, int* eoe ){
  unsigned int noise = random_int( -1, 1 );
  int iControl;
  switch( action ){
  case LEFT:
    iControl = -50 + noise;
    break;
  case NONE:
    iControl = 0 + noise;
    break;
  case RIGHT:
    iControl = 50 + noise;
    break;
  }
  double g = 9.8;
  double m = 2.0;
  double M = 8.0;
  double l = 0.50;
  double alpha = 1./(m+M);
  double step = 0.1;
  double control = (double)iControl;
  double acceleration = 
    (g*sin(state_p) - 
     alpha*m*l*pow(state_v,2)*sin(2*state_p)/2. - 
     alpha*cos(state_p)*control) / 
    (4.*l/3. - alpha*m*l*pow(cos(state_p),2));
  *next_state_p = state_p +state_v*step;
  *next_state_v = state_v + acceleration*step;
  if( *next_state_p > PI/2. || *next_state_p < -PI/2. ){
    *reward = -1;
    *eoe = 0; 
  }else{
    *reward = 0;
    *eoe = 1;
  }
}
