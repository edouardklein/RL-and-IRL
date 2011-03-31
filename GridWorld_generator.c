#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <gsl/gsl_matrix.h>
#include "utils.h"
#include "GridWorld.h"
#define NUMBER_OF_WALKS 500
#define MAX_WALK_LENGTH  (GRID_HEIGHT+GRID_WIDTH)

int main(void){
  srand(time(NULL)+getpid()); rand(); rand();rand();
  for( unsigned int i = 0 ; i < NUMBER_OF_WALKS ; i++ ){
    unsigned int state_x = random_int( 1, GRID_WIDTH );
    unsigned int state_y = random_int( 1, GRID_HEIGHT );
    int eoe = 1;
    for( unsigned int j = 0 ; j < MAX_WALK_LENGTH && eoe == 1 ; 
	 j++ ){
      if( j == MAX_WALK_LENGTH - 1 ){
	eoe = 0;
      }
      unsigned int next_state_x = state_x;
      unsigned int next_state_y = state_y;
      unsigned int action = random_int( 1, 4 );
      unsigned int true_action = action;
      int is_noisy = rand_1_in_10();
      if( is_noisy ){
	true_action = random_int( 1, 4 );
      }
      switch( true_action ){
      case UP:
	next_state_y++;
	if( next_state_y > GRID_HEIGHT ){
	  next_state_y = GRID_HEIGHT;
	}
	break;
      case DOWN:
	next_state_y--;
	if( next_state_y < 1 ){
	  next_state_y = 1;
	}
	break;
      case RIGHT:
	next_state_x++;
	if( next_state_x > GRID_WIDTH ){
	  next_state_x = GRID_WIDTH;
	}
	break;
      case LEFT:
	next_state_x--;
	if( next_state_x < 1 ){
	  next_state_x = 1;
	}
	break;
      }
      int reward = 0;
      if( state_x == GRID_WIDTH && 
	  state_y == GRID_HEIGHT ){
	reward = 1;
	eoe = 0;
      }
      printf("%d %d %d %d %d %d %d\n",
	     state_x, state_y, action, 
	     next_state_x, next_state_y, reward, eoe );
      state_x = next_state_x;
      state_y = next_state_y;
    }
  }
}
