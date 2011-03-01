/*
  Generate a random walk in a 4 state chain.
  See \cite{lagoudakis2003least}.
  Output on the standard output a transition file
  of the following format :
  s a s' r eoe
  s and s' are ints in [1:4]
  a is an int in [0:1], 0 is left and 1 is right
  r is the reward, 1 in states 2 and 3, 0 otherwise
  eoe is the end of episode flag. 1 if not the EndOfEpisode
  0 if EndOfEpisode
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

/* Return an int in [1:4] */
int rand_4(){
  return rand()%4 + 1;
}

/* Return an int in [0:1] */
int rand_2(){
  return rand()%2;
}

/* Return 0 w.p. 0.9 and 1 w.p. 0.1 */
int rand_1_in_10(){
  if( (double)rand() > (double)RAND_MAX*0.1 ){
    return 0;
  }
  return 1;
}

int main(int argc, char* argv[]){
  srand(time(NULL)+getpid()); rand(); rand();rand();
  if (argc != 2){
    printf("usage :\n %s <trajLength>\n", argv[0] );
    return -1;
  }
  unsigned int trajLength = atoi(argv[1]);
  
  int s = rand_4();

  for( unsigned int j = 0 ; j < trajLength ; j++ ){
    int a = rand_2();
    int is_noisy = rand_1_in_10();
    int true_a = a;
    if( is_noisy ){
      true_a = !a;
    }
    int s_dash = s;
    switch( true_a ){
    case 0: //Left
      if( s > 1 ){
	s_dash--;
      }
      break;
    case 1: //Right
      if( s < 4 ){
	s_dash++;
      }
      break;
    }
    int reward = 0;
    if( s == 2 || s == 3 ){
      reward = 1;
    }
    int eoe = 1;
    printf( "%d %d %d %d %d\n",
	    s, a, s_dash, reward, eoe );
    s = s_dash;
  }
  
  return 0;
}
