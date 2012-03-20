
gsl_matrix* proj_lstd_lspi_ANIRL( gsl_matrix* D_E, 
				  gsl_matrix* D );

gsl_matrix* lstd_mu_omega( gsl_matrix* D_mu, 
			   gsl_matrix* (*pi)(gsl_matrix*) );

gsl_matrix* lstd_mu_op_omega(  gsl_matrix* D_mu );
