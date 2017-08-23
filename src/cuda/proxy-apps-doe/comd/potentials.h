/*
  Various implementations of the following potentials:
    1) LJ 
    2) EAM - original unoptimized version (cubic splines)
    3) EAM - optimized version (cubic splines)
    4) EAM - experimental version (chebyshev polynomials)
*/

__device__
void ljForce(real_t r, real_t *f)
{
}

__device__
void eamInterpolateDeriv(real_t r, const real_t *values, int n_values, real_t *value1, real_t *f1)
{
    int i1;
    real_t gi, gi1;

    // extract values from potential 'struct'
    real_t x0 = values[n_values + 3];
    real_t xn = values[n_values + 4];
    real_t invDx = values[n_values + 5];

    // identical to Sriram's loop in eam.c
    if (r < x0) r = x0;
    else if (r > xn) r = xn;

    r = (r - x0) * (invDx) ;
    i1 = (int)floor(r);

    /* reset r to fractional distance */
    r = r - floor(r);

    gi  = values[i1 + 2] - values[i1];
    gi1 = values[i1 + 3] - values[i1 + 1];

    // values[i1-1] is guaranteed(?) inbounds because 
    // a->x0 = x0 + (xn-x0)/(double)n; 
    // appears in allocPotentialArray
    *value1 = values[i1 + 1] + 0.5 * r * (r * (values[i1 + 2] + values[i1] -2.0 * values[i1 + 1]) + gi);
    if (i1 <= 1) *f1 = 0.0;
      else *f1 = 0.5 * (gi + r * (gi1 - gi)) * invDx;
}

__inline__ __device__
void eamInterpolateDeriv_opt(real_t r, const real_t *values, const real_t x0, const real_t xn, const real_t invDx, real_t &value1, real_t &f1)
{
    int i1;
    real_t gi, gi1;

    // identical to Sriram's loop in eam.c
    r = max(r, x0);
    r = min(r, xn);

    r = (r - x0) * (invDx);
    i1 = (int)floor(r);

    /* reset r to fractional distance */
    r = r - i1;

    // using LDG on Kepler only
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350  
    real_t v0 = __ldg(values + i1);
    real_t v1 = __ldg(values + i1 + 1);
    real_t v2 = __ldg(values + i1 + 2);
    real_t v3 = __ldg(values + i1 + 3);
#else
    real_t v0 = values[i1];
    real_t v1 = values[i1 + 1];
    real_t v2 = values[i1 + 2];
    real_t v3 = values[i1 + 3];
#endif

    gi  = v2 - v0;
    gi1 = v3 - v1;

    // write result
    value1 = v1 + (real_t)0.5 * r * (r * (v2 + v0 - 2 * v1) + gi);
    if (i1 <= 1) f1 = 0;
      else f1 = (real_t)0.5 * (gi + r * (gi1 - gi)) * invDx;
}

__device__
real_t chebev(const potentialarray_t cheb, real_t x)
{
   real_t a = cheb.x0;
   real_t b = cheb.xn;
   real_t *c = cheb.values;
   int m = cheb.n;

   real_t d = 0;
   real_t dd = 0;
   real_t y = (2 * x - a - b) / (b - a);
   real_t y2 = 2 * y;

   for (int j = m-1; j > 0; j--) {
      real_t sv = d;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350  
      d = y2 * d - dd + __ldg(c + j);
#else
      d = y2 * d - dd + c[j];
#endif
      dd = sv;
   }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350  
   real_t ch = y * d - dd + (real_t)0.5 * __ldg(c + 0);
#else
   real_t ch = y * d - dd + (real_t)0.5 * c[0];
#endif
   return ch;
}

