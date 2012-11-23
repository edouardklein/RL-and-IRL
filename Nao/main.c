#include <stdio.h>

#include <stdlib.h>

#include <gsl/gsl_matrix.h>
#include "LSTDQ.h"
#include "LSTDmu.h"
#include "utils.h"
#include "greedy.h"
#include "LSPI.h"
#include "RL_Globals.h"
#include "phipsi.h"

#define SAS     "sas.txt"
#define OMEG0   "omeg0.txt"
#define OMEG   "omeg.txt"
#define S     "s.txt"
#define S0     "s0.txt"
#define A   "a.txt"

#define DIM 3   //nb d'articulations consideree
#define LONG    10  //longueur d'une trajectoire
#define GAMMA   0.9     //gamma de l'irl
#define K   15  //nb de fct de mapping
#define EPSILON 0.001   //TBD : arret de l'algo

gsl_matrix* g_mActions = NULL;

gsl_matrix* initial_state( void );

unsigned int g_iS = 3;
unsigned int g_iA = 3;
unsigned int g_iIt_max_lspi = 50;
gsl_matrix* (*g_fPhi)(gsl_matrix*) = &phi;
gsl_matrix* g_mOmega = NULL;
double g_dLambda_lstdQ = 0.1;
double g_dGamma_lstdq =  0.9;
double g_dEpsilon_lspi = 0.01;
double g_dLambda_lstdmu = 0.1;
double g_dGamma_anirl = 0.9;
double g_dEpsilon_anirl = 0.01;
unsigned int g_iIt_max_anirl = 2;
//gsl_matrix* g_mActions = NULL;
gsl_matrix* (*g_fPsi)(gsl_matrix*) = &psi;
gsl_matrix* (*g_fSimulator)(int) = NULL;
gsl_matrix* (*g_fS_0)(void) = &initial_state;
unsigned int g_iMax_episode_len = 20; //Shorter episodes are



gsl_matrix* initial_state( void ){
  gsl_matrix* answer = file2matrix("s0.txt",3);

  return answer;
}


int main()

{

    int i=0,j=0;

    printf("Hello world!\n");




g_mActions = file2matrix( A, g_iA );

gsl_matrix* mat_sas = file2matrix(SAS, (2*g_iS+g_iA + 1 + 1)); //DIM-s DIM-a DIM-s' 1-42 1-e
gsl_matrix* mat_lstd =proj_lstd_lspi_ANIRL(mat_sas,mat_sas);
//gsl_matrix* mat_omega = file2matrix(OMEG, g_iK);

 gsl_matrix_fprintf(stdout,mat_lstd,"%e ");

/* for(i=0;i<mat_lstd->size1;i++) */
/*     { */
/*      for(j=0;j<mat_lstd->size2;j++) */
/*      { */
/*          printf("%f ",gsl_matrix_get(mat_lstd, i, j)); */

/*      } */
/*      printf("\n"); */
/*     } */


    return 0;

}

