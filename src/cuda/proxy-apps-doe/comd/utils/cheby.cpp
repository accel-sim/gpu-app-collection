#include "cheby.h"


// Based on the OCL version
//different call signature than the regular version
static void eamInterpolateDerivlocal(real_t r,
	real_t* values,
	int n_values,
	real_t *range,
	real_t *value1, 
	real_t *f1)
{
    int i1;
    int i;
    real_t gi, gi1;

    // using different data struct here
    real_t x0 = range[0];
    real_t xn = range[1];
    real_t invDx = range[2];

    // identical to Sriram's loop in eam.c
    if ( r<x0) r = x0;
    else if (r>xn) r = xn;

    r = (r-x0)*(invDx) ;
    i1 = (int)floor(r);

    /* reset r to fractional distance */
    r = r - floor(r);

    gi  = values[i1+1] - values[i1-1];
    gi1 = values[i1+2] - values[i1];


    // values[i1-1] is guaranteed(?) inbounds because 
    // a->x0 = x0 + (xn-x0)/(double)n; 
    // appears in allocPotentialArray
    *value1 = values[i1] + 0.5*r*(
	    r*(values[i1+1]+ values[i1-1] -2.0*values[i1]) +
	    gi
	    );
    if (*value1 > 1.0e10) *value1 = 0.0;
    if(i1<=0) 
	*f1 = 0.0;
    else 
	*f1 = 0.5*(gi + r*(gi1-gi))*invDx;

    return;

}

static inline void eamInterpolateDeriv(struct potentialarray_t *a, real_t r, int iType, int jType, real_t *value1, real_t *f1) {
  /**
   *
   * This routine will not crash if r is out of range.
   *
   * if ( r < a->x0) r = a->x0;
   * if ( r > a->xn)   r = a->xn;
   **/
   
  int i1;
  real_t gi, gi1;

  if ( r<a->x0) r = a->x0;
  else if (r>a->xn) r = a->xn;
  
  r = (r-a->x0)*(a->invDx) ;
  i1 = (int)floor(r);

  /* reset r to fractional distance */
  r = r - floor(r);

  gi  = a->values[i1+1] - a->values[i1-1];
  gi1 = a->values[i1+2] - a->values[i1];


  *value1 = a->values[i1] + 0.5*r*(
				   r*(a->values[i1+1]+ a->values[i1-1] -2.0*a->values[i1]) +
				   gi
				   );
  if(i1<=0) *f1 = 0.0;
  else *f1 = 0.5*(gi + r*(gi1-gi))*a->invDx;

  return;
}

/** Given an EAM potential, generate the corresponding Chebychev approximation 
 * with n coefficients **/
struct eam_cheby_t *setChebPot(eampotential_t *pot, int n)
{
    printf("Generating Chebychev coefficients: ");

    struct eam_cheby_t *retCheb;

    retCheb = (eam_cheby_t*)malloc(sizeof(eam_cheby_t));

    printf("phi, ");
    retCheb->phi = genCheb(pot->phi, n);
    retCheb->dphi = genDer(retCheb->phi);

    printf("rho, ");
    retCheb->rho = genCheb(pot->rho, n);
    retCheb->drho = genDer(retCheb->rho);

    printf("f");
    retCheb->f = genCheb(pot->f, n);
    retCheb->df = genDer(retCheb->f);

    printf("\n");
    return retCheb;
}

/** Given a tabulated potential array, generate the first n coefficients of 
 * the corresponding Chebychev approximation **/
struct potentialarray_t *genCheb(potentialarray_t *pot, int n)
{
    potentialarray_t *ch_pot = (struct potentialarray_t*)malloc(sizeof(struct potentialarray_t));
    ch_pot->x0 = pot->x0;
    ch_pot->xn = pot->xn;
    ch_pot->n = n;
    ch_pot->values = (real_t*)malloc((n)*sizeof(real_t));

    real_t a = pot->x0; 
    real_t b = pot->xn; 
    real_t *c = ch_pot->values; 
    real_t *values = pot->values;
    int n_values = pot->n;
    real_t invDx = pot->invDx;

#if (DIAG_CHEBY > 0)
    printf("range is "EMT1", "EMT1", "EMT1"\n", a, b, invDx);
    printf("n_values = %d\n", n_values);
#endif

    int k,j;
    real_t fac,bpa,bma;
    real_t r_dummy;

    real_t range[3];
    range[0] = a;
    range[1] = b;
    range[2] = invDx;

    real_t *f = new real_t[n];
    bma=0.5*(b-a);
    bpa=0.5*(b+a);
    for (k=0;k<n;k++) {
	real_t y=cos(PI*(k+0.5)/n);
#if (DIAG_CHEBY > 0)
	eamInterpolateDeriv(pot, y*bma+bpa, 0, 0, &f[k], &r_dummy);
	printf("old %d, "EMT1", "EMT1"\n", k, y*bma+bpa, f[k]);
#endif
	eamInterpolateDerivlocal(y*bma+bpa, values, n_values, range, &f[k], &r_dummy);
#if (DIAG_CHEBY > 0)
	printf("new %d, "EMT1", "EMT1"\n", k, y*bma+bpa, f[k]);
#endif
    }
    fac=2.0/n;
    for (j=0;j<n;j++) {
	double sum=0.0;
	for (k=0;k<n;k++)
	    sum += f[k]*cos(PI*j*(k+0.5)/n);
	c[j]=fac*sum;
#if (DIAG_CHEBY > 0)
	printf("%d, "EMT1"\n",j, c[j]);
#endif
    }
	delete f;
    return ch_pot;
}

/** given n Chebychev coefficients for some F, compute the corresponding coefficients for dF **/
void chder(real_t a, real_t b, real_t *c, real_t *cder, int n)
{
#if (DIAG_CHEBY > 0)
    printf("a = %e,  b = %e\n", a, b);
#endif
    int j;
    double con;
#if (DIAG_CHEBY > 0)
    printf("cder:\n");
#endif
    cder[n-1]=0.0;
#if (DIAG_CHEBY > 0)
    printf("%d, %e\n", n-1, cder[n-1]);
#endif
    cder[n-2]=2.0*(n-1)*c[n-1];
#if (DIAG_CHEBY > 0)
    printf("%d, %e\n", n-2, cder[n-2]);
#endif
    for(j=n-2;j>0;j--) {
	cder[j-1]=cder[j+1]+2*(j)*c[j];
#if (DIAG_CHEBY > 0)
        printf("%d, %e\n", j-1, cder[j-1]);
#endif
    }
    con=2.0/(b-a);
#if (DIAG_CHEBY > 0)
    printf("con: %e\n", con);
#endif
    for (j=0;j<n;j++) {
	cder[j]=cder[j]*con;
#if (DIAG_CHEBY > 0)
        printf("%d, %e, %e\n", j, c[j], cder[j]);
#endif
    }
}

/** Given a Chebychev potential array, generate the corresponding array for the derivative **/
struct potentialarray_t *genDer(potentialarray_t *ch)
{
    real_t a = ch->x0;
    real_t b = ch->xn;
    real_t* c = ch->values;
    int n = ch->n;

    potentialarray_t *chder = (struct potentialarray_t*)malloc(sizeof(struct potentialarray_t));
    chder->x0 = a;
    chder->xn = b;
    chder->invDx = 1.0;
    chder->n = n;
    chder->values = (real_t*)malloc(n*sizeof(real_t));

    real_t* cder = chder->values;

#if (DIAG_CHEBY > 0)
    printf("a = %e,  b = %e\n", a, b);
    printf("n = %d\n", n);
#endif
    int j;
    double con;
#if (DIAG_CHEBY > 0)
    printf("cder:\n");
#endif
    cder[n-1]=0.0;
#if (DIAG_CHEBY > 0)
    printf("%d, %e\n", n-1, cder[n-1]);
    fflush(stdout);
#endif
    cder[n-2]=2.0*(n-1)*c[n-1];
#if (DIAG_CHEBY > 0)
    printf("%d, %e\n", n-2, cder[n-2]);
#endif
    for(j=n-2;j>0;j--) {
	cder[j-1]=cder[j+1]+2*(j)*c[j];
#if (DIAG_CHEBY > 0)
        printf("%d, %e\n", j-1, cder[j-1]);
#endif
    }
    con=2.0/(b-a);
#if (DIAG_CHEBY > 0)
    printf("con: %e\n", con);
#endif
    for (j=0;j<n;j++) {
	cder[j]=cder[j]*con;
#if (DIAG_CHEBY > 0)
        printf("%d, %e, %e\n", j, c[j], cder[j]);
#endif
    }
    return chder;
}

/** given a list of Chebyshev coefficients c, compute the value at x
 * x must be in the range [a, b]
 * Modified the call signature to use the potentialarray_t
 * which incorporates a, b, c as x0, xn, values **/
real_t eamCheb(potentialarray_t *cheb, real_t x) 
{
    real_t a = cheb->x0;
    real_t b = cheb->xn;
    real_t *c = cheb->values;
    int m = cheb->n;
    int i;

#if (DIAG_CHEBY > 1)
    printf("x0 = "EMT1", xn = "EMT1", n = %d \n", a, b, m);
    for (i=0;i<m;i++) printf(EMT1"\n", cheb->values[i]);
#endif

    real_t d, dd, sv, y, y2;
    real_t ch;
    int j;
    if (x < a) x = a;
    if (x > b) b = b;
    /*
    if ((x-a)*(x-b) > 0.0) {
        printf("x not in range in eamCheb, %f\n", x);
    }
    */
    d=0.0;
    dd=0.0;
    y=(2.0*x-a-b)/(b-a);
    y2=2.0*y;
    for(j=m-1;j>0;j--) {
        sv=d;
        d=y2*d-dd+c[j];
        dd=sv;
    }
    ch=y*d-dd+0.5*c[0];
    return ch;
}


/** given a list of Chebyshev coefficients c, compute the value at x 
 * x must be in the range [a, b] **/
real_t chebev(real_t a, real_t b, real_t *c,int m, real_t x) 
{
real_t d, dd, sv, y, y2;
real_t ch;
int j;
if ((x-a)*(x-b) > 0.0) {
printf("x not in range in chebev, %f\n", x);
}
d=0.0;
dd=0.0;
y=(2.0*x-a-b)/(b-a);
y2=2.0*y;
for(j=m-1;j>0;j--) {
sv=d;
d=y2*d-dd+c[j];
dd=sv;
}
ch=y*d-dd+0.5*c[0];
return ch;
}

