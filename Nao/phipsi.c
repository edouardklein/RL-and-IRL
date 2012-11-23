#include <gsl/gsl_matrix.h>
#include "RL_Globals.h"
#include <math.h>
#ifndef M_PI
#define M_PI           3.14159265358979323846
#endif



unsigned int g_iK = (10*64); /* dim(\phi) */
unsigned int g_iP = (64); /* dim(\psi) */

int sigma_map=1600;


float centroids[64][3]= {1251,    -1317.6,    318.8,
1251,    -1317.6,    629.6,
1251,    -1317.6,    940.4,
1251,    -1317.6,    1251,.200,
1251,    -986.20,    318.8,
1251,    -986.20,    629.6,
1251,    -986.20,    940.4,
1251,    -986.20,    1251,.200,
1251,    -654.80,    318.8,
1251,    -654.80,    629.6,
1251,    -654.80,    940.4,
1251,    -654.80,    1251,.200,
1251,    -323.40,    318.8,
1251,    -323.40,    629.6,
1251,    -323.40,    940.4,
1251,    -323.40,    1251,.200,
417,    -1317.6,    318.8,
417,    -1317.6,    629.6,
417,    -1317.6,    940.4,
417,    -1317.6,    1251,.200,
417,    -986.20,    318.8,
417,    -986.20,    629.6,
417,    -986.20,    940.4,
417,    -986.20,    1251,.200,
417,    -654.80,    318.8,
417,    -654.80,    629.6,
417,    -654.80,    940.4,
417,    -654.80,    1251,.200,
417,    -323.40,    318.8,
417,    -323.40,    629.6,
417,    -323.40,    940.4,
417,    -323.40,    1251,.200,
-417,    -1317.6,    318.8,
-417,    -1317.6,    629.6,
-417,    -1317.6,    940.4,
-417,    -1317.6,    1251,.200,
-417,    -986.20,    318.8,
-417,    -986.20,    629.6,
-417,    -986.20,    940.4,
-417,    -986.20,    1251,.200,
-417,    -654.80,    318.8,
-417,    -654.80,    629.6,
-417,    -654.80,    940.4,
-417,    -654.80,    1251,.200,
-417,    -323.40,    318.8,
-417,    -323.40,    629.6,
-417,    -323.40,    940.4,
-417,    -323.40,    1251,.200,
-1251,    -1317.6,    318.8,
-1251,    -1317.6,    629.6,
-1251,    -1317.6,    940.4,
-1251,    -1317.6,    1251,.200,
-1251,    -986.20,    318.8,
-1251,    -986.20,    629.6,
-1251,    -986.20,    940.4,
-1251,    -986.20,    1251,.200,
-1251,    -654.80,    318.8,
-1251,    -654.80,    629.6,
-1251,    -654.80,    940.4,
-1251,    -654.80,    1251,.200,
-1251,    -323.40,    318.8,
-1251,    -323.40,    629.6,
-1251,    -323.40,    940.4,
-1251,    -323.40,    1251,.200
};



gsl_matrix* psi( gsl_matrix* s ){
  gsl_matrix* answer = gsl_matrix_calloc( g_iP, 1 );
  double x,y,z;
  float temp=0;
  double  res=0;

  for (int k=0;k<g_iP;k++){
    //for (int j = 0;j<g_iS;j++){
      x= gsl_matrix_get( s, 0, 0 );
      y= gsl_matrix_get( s, 0, 1 );
      z= gsl_matrix_get( s, 0, 2 );

      //calcul avec la gaussienne pour chaque dim
      temp=exp( - (float)(pow((x-centroids[k][0]),2) + pow((y-centroids[k][1]),2) + pow((z-centroids[k][2]),2))
                    / (float)(2.*sigma_map*sigma_map) )
                    / (float)(sigma_map * sqrt(2.*M_PI));

  /*  printf("%f\t%f\n",x,centroids[k][0]);
    printf("%f\t%f\n",y,centroids[k][1]);
    printf("%f\t%f\n",z,centroids[k][2]);
    float   a=pow((x-centroids[k][0]),2) + pow((y-centroids[k][1]),2) + pow((z-centroids[k][2]),2);
    printf("%f\n",a);
    float b=(a/(float)(2*sigma_map*sigma_map));
    printf("%f\n",b);
    float c=exp(-b);
    printf("%f\n",c);
    printf("%f\n\n",temp);*/

        gsl_matrix_set( answer, k, 0, temp);
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
        if(gsl_matrix_get( a, 0, 0 )==gsl_matrix_get(g_mActions,i,0)
           &&gsl_matrix_get( a, 0, 1 )==gsl_matrix_get(g_mActions,i,1)
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
