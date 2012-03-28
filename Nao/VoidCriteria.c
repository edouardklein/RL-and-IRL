//FIXME:Corriger ANIRL versionLSTDmu pour ne plus nécessiter ce qui se trouve dans ce fichier.

#include <gsl/gsl_matrix.h>

double g_dBest_error = -1;
double g_dBest_true_error = -1;
double g_dBest_diff = -1;
double g_dBest_t = 0;
gsl_matrix* g_mBest_omega = NULL;


double true_diff_norm( gsl_matrix* r){
  return -1;
}

double true_V_diff( gsl_matrix* r){
  return -1;
}
