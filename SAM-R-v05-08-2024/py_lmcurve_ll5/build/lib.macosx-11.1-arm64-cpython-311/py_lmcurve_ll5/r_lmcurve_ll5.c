/*

Binding of lmcurve/lmfit by Joachim Wuttke, Forschungszentrum Juelich GmbH for the R programming language.
See also COPYING and README.txt

Author: Marco Foit

*/

#include <math.h>
#include <stdio.h>


#include "lmstruct.h"
#include "lmmin.h"
#include "lmcurve_user.h"
#include "lmmin.c"
#include "lmcurve_user.c"

typedef struct Context Context;
struct Context {
  double b;
  double c;
  double d;
  double e;
  double f;
};

double N(double C, const double *p, const void *user_data) {
	Context* ctx = (Context*)user_data;

	int pi = 0;
	double b = isnan(ctx->b) ? p[pi++] : ctx->b;
	double c = isnan(ctx->c) ? p[pi++] : ctx->c;
	double d = isnan(ctx->d) ? p[pi++] : ctx->d;
	double e = isnan(ctx->e) ? p[pi++] : ctx->e;
	double f = isnan(ctx->f) ? p[pi++] : ctx->f;

	return c + (d-c) / pow(1 + pow(C/e,b), f);
}

void r_lmcurve_ll5(double* x, double* y, int* n, double* b, double* c, double* d, double* e, double* f) {
	Context ctx;
	ctx.b = b[0];
	ctx.c = c[0];
	ctx.d = d[0];
	ctx.e = e[0];
	ctx.f = f[0];

    int pn = 0; /* number of parameters in model function f */

    if(isnan(ctx.b)) pn++;
    if(isnan(ctx.c)) pn++;
    if(isnan(ctx.d)) pn++;
    if(isnan(ctx.e)) pn++;
    if(isnan(ctx.f)) pn++;


	double par[5] = { 1, 1, 1, 1, 1}; /* really bad starting value */

    // !!! DATA-FIX, force plateau at x=0
   	// conc = sapply(conc, function(c) { if(c) c else 10^(-100) })
   	for(int i=0; i<n[0]; i++) {
   		if(x[i] == 0.0)
   			x[i] = 1.e-100;
   	}


    lm_control_struct control = lm_control_double;

   	control.stepbound = 1.e-12;
   	control.patience = 1.e+4;


    lm_status_struct status;
    lmcurve_user( pn, par, n[0], x, y, N, &ctx, &control, &status );

    // TODO actually we do not know exactly what the status means here
 	int has_error;
 	switch(status.outcome) {
 		case 9:
 		case 10:
 		case 11:
 			has_error = 1;
		break;
		default:
 			has_error = 0;
		break;
 	}

	if(!has_error) {
		int pi = 0;
		if(isnan(ctx.b)) b[0] = par[pi++];
		if(isnan(ctx.c)) c[0] = par[pi++];
		if(isnan(ctx.d)) d[0] = par[pi++];
		if(isnan(ctx.e)) e[0] = par[pi++];
		if(isnan(ctx.f)) f[0] = par[pi++];
	}
}

