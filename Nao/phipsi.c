#include <gsl/gsl_matrix.h>
#include "RL_Globals.h"
#include <math.h>
#include "InvertedPendulum.h"


unsigned int g_iK = (10*15); /* dim(\phi) */
unsigned int g_iP = (15); /* dim(\psi) */

int sigma_map=2000;


int centroids[15][3]=
{0,-276,254,\
0,-276,1300,\
0,-828,777,\
0,-1380,254,\
0,-1380,1300,\
-695,-276,254,\
-695,-276,1300,\
-695,-828,777,\
-695,-1380,254,\
-695,-1380,1300,\
695,-276,254,\
695,-276,1300,\
695,-828,777,\
695,-1380,254,\
695,-1380,1300};



gsl_matrix* psi( gsl_matrix* s ){
  gsl_matrix* answer = gsl_matrix_calloc( g_iP, 1 );
  double x;

  unsigned int index = 0;

double temp[g_iS];
double  res=0;

    for (int k=0;k<g_iP;k++)
    {
        for (int j = 0;j<g_iS;j++)
        {
            x= gsl_matrix_get( s, 0, j );

                //calcul avec la gaussienne pour chaque dim
                temp[j]=exp(- (x-centroids[k][j])*((x-centroids[k][j])) / (2*sigma_map*sigma_map) ) / (sigma_map * sqrt(2*PI));
                temp[j]=pow(temp[j],2);
        }

        for (int j = 0;j<g_iS;j++)
        {
            res+=temp[j];
        }

        gsl_matrix_set( answer, k, 0, sqrt(res));
    }

  return answer;
}



gsl_matrix* phi( gsl_matrix* sa ){

    gsl_matrix* answer = gsl_matrix_calloc( g_iK, 1 );

    gsl_matrix* a = gsl_matrix_calloc( 1,g_iA );
    gsl_matrix* s = gsl_matrix_calloc( 1,g_iS );

    gsl_matrix_set(s,0,0,gsl_matrix_get(sa,0,0));
    gsl_matrix_set(s,0,1,gsl_matrix_get(sa,0,1));
    gsl_matrix_set(s,0,2,gsl_matrix_get(sa,0,2));
    gsl_matrix_set(a,0,0,gsl_matrix_get(sa,0,3));
    gsl_matrix_set(a,0,1,gsl_matrix_get(sa,0,4));
    gsl_matrix_set(a,0,2,gsl_matrix_get(sa,0,5));

    int i=0;
    int j=0;

    gsl_matrix* temp=psi( s );

    for(i =0;i<(g_iK/g_iP);i++)//on cherche quel action a été lue
    {
        if(gsl_matrix_get( a, 0, 0 )==gsl_matrix_get(g_mActions,i,0)\
           &&gsl_matrix_get( a, 0, 1 )==gsl_matrix_get(g_mActions,i,1)\
           &&gsl_matrix_get( a, 0, 2 )==gsl_matrix_get(g_mActions,i,2))
            break;
    }

    if(i==g_iK/g_iP)//si on a fait la boucle en entier
    {
        fprintf(stderr,"problem occured in phipsi: unknown a\n");

        for(int n=0;n<a->size1;n++)
            {
             for(int  m=0;m<a->size2;m++)
             {
                 printf("%f ",gsl_matrix_get(a, n, m));

             }
             printf("\n");
            }

        for(int n=0;n<sa->size1;n++)
            {
             for(int  m=0;m<sa->size2;m++)
             {
                 printf("%f ",gsl_matrix_get(sa, n, m));

             }
             printf("\n");
            }

        exit(1);
    }


    for(int k=0;k<g_iK;k++)//des zeros partout
    {
        gsl_matrix_set( answer, k, 0, 0 );
    }

    for(j=0;j<g_iP;j++)//le psi requis
    {
        gsl_matrix_set( answer, (i*g_iP)+j, 0, gsl_matrix_get(temp,j,0) );
    }

gsl_matrix_free( a );
gsl_matrix_free( s );


  return answer;
}
