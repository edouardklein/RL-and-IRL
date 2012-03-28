#include <stdio.h>
#include <stdlib.h>

#include <gsl/gsl_matrix.h>
#include "LSTDQ.h"
#include "LSTDmu.h"
#include "utils.h"
#include "greedy.h"
#include "LSPI.h"
#include "RL_Globals.h"

#define SAS     "../resources/sas.txt"
#define OMEG0   "../resources/omeg0.txt"
#define OMEG   "../resources/omeg.txt"
#define S     "../resources/s.txt"
#define S0     "../resources/s0.txt"
#define A   "../resources/a.txt"

#define DIM 3   //nb d'articulations consideree
#define LONG    10  //longueur d'une trajectoire
#define GAMMA   0.9     //gamma de l'irl
#define K   15  //nb de fct de mapping
#define EPSILON 0.001   //TBD : arret de l'algo

gsl_matrix* g_mActions = NULL;

int main()
{
    int i=0,j=0;

    printf("Hello world!\n");




g_mActions = file2matrix( A, g_iA );

gsl_matrix* mat_sas = file2matrix(SAS, (2*g_iS+g_iA + 1 + 1)); //DIM-s DIM-a DIM-s' 1-42 1-e
gsl_matrix* mat_lstd =proj_lstd_lspi_ANIRL(mat_sas,mat_sas);
//gsl_matrix* mat_omega = file2matrix(OMEG, g_iK);

for(i=0;i<mat_lstd->size1;i++)
    {
     for(j=0;j<mat_lstd->size2;j++)
     {
         printf("%f ",gsl_matrix_get(mat_lstd, i, j));

     }
     printf("\n");
    }


    return 0;
}
