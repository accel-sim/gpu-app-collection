#include <stdio.h>

/*
#define TRACE_DX
#define TRACE_DZ
#define TRACE_DIFFLUX_UX
#define TRACE_DIFFLUX_UY
#define TRACE_DIFFLUX_UZ
#define TRACE_DIFFLUX_E
#define TRACE_UPDATE_RHO
#define TRACE_UPDATE_E
#define TRACE_UPDATE_U
#define TRACE_UPDATE_U_ALT
*/

#define RADIUS 4
#define TYPE  double

#define CACHE_LINE_SIZE  128

#ifdef UNROLL
	#define OUTER_UNROLL	#pragma unroll 9
#else
	#define OUTER_UNROLL	//#pragma unroll 9
#endif

#ifndef TILE_DIMX
    #define TILE_DIMX 32
#endif

#ifndef TILE_DIMY
    #define TILE_DIMY 8
#endif

#ifndef LDG
    #define __ldg( X )  (*(X))
#endif

__constant__ TYPE c_first[RADIUS+1];
__constant__ TYPE c_second[RADIUS+1];

#define	ONE_THIRD	(1./3.)
#define TWO_THIRDS  (2./3.)
#define	FOUR_THIRDS	(4./3.)

void process_error( const cudaError_t &error, char *string=0, bool verbose=false )
{
    if( error != cudaSuccess || verbose )
    {
        int current_gpu = -1;
        cudaGetDevice( &current_gpu );

        printf( "GPU %d: ", current_gpu );
        if( string )
            printf( string );
        printf( ": %s\n", cudaGetErrorString( error ) );
    }

    if( error != cudaSuccess )
        exit(-1);
}


inline __device__ void advance( TYPE *field, const int num_points )
{
    #pragma unroll
    for(int i=0; i<num_points-1; i++)
        field[i] = field[i+1];
}

#ifdef TRACE_UPDATE_RHO
template<int tile_dimx, int tile_dimy, int radius>
__global__ void update_rho_single_pass( 
                  TYPE* g_rho, 
	              const TYPE* g_ux, const TYPE* g_uy, const TYPE* g_uz,
	              const TYPE delta_t,
	              const int nx, const int ny, const int nz,
                  const int nx_pad, const int ny_pad, const int nz_pad )
{
	int ix = blockIdx.x * tile_dimx + threadIdx.x;
	int iy = blockIdx.y * tile_dimy + threadIdx.y;

	int slice_stride = nx_pad * ny_pad;

	int idx_out = iy * nx_pad + ix;
	int idx_in  = idx_out - radius * slice_stride;

	const int diameter = 2*radius + 1;

	TYPE queue[2*radius+1];

	#pragma unroll
	for( int i=1; i<diameter; i++ )
	{
		queue[i] = __ldg( &g_uz[idx_in] );
		idx_in += slice_stride;
	}

	OUTER_UNROLL
	for( int iz=0; iz<nz; iz++ )
	{
		advance( queue, diameter );
		queue[diameter-1] = __ldg( &g_uz[idx_in] );

		TYPE del_momenta = 0;
				
		#pragma unroll
		for( int i=1; i<=radius; i++ )
		{
			del_momenta += c_first[i] * (   __ldg( &g_ux[idx_out+i] )        - __ldg( &g_ux[idx_out-i] )
			                              + __ldg( &g_uy[idx_out+i*nx_pad] ) - __ldg( &g_uy[idx_out-i*nx_pad] )
			                              + queue[radius+i]                  - queue[radius-i]         );
		}

		g_rho[idx_out] = __ldg( &g_rho[idx_out] ) - delta_t * del_momenta;

		idx_in  += slice_stride;
		idx_out += slice_stride;
	}
}
#endif

#ifdef TRACE_DX
template<int tile_dimx, int tile_dimy, int radius>
__global__ void first_derivative_x( TYPE* g_deriv, TYPE *g_field, 
									const int nx, const int ny, const int nz,
									const int nx_pad, const int ny_pad, const int nz_pad )
{
	
	int ix = blockIdx.x * tile_dimx + threadIdx.x;
	int iy = blockIdx.y * tile_dimy + threadIdx.y;
	int stride = nx_pad * ny_pad;

	int idx = iy * nx_pad + ix;

	OUTER_UNROLL
	for( int iz=0; iz<nz; iz++ )
	{
		TYPE deriv = c_first[0] * __ldg( &g_field[idx] );
		#pragma unroll
		for( int i=1; i<=radius; i++ )
			deriv += c_first[i] * ( __ldg( &g_field[idx+i] ) - __ldg( &g_field[idx-i] ) );

		g_deriv[idx] = deriv;

		idx += stride;
	}
}
#endif


#ifdef TRACE_DZ
template<int tile_dimx, int tile_dimy, int radius>
__global__ void first_derivative_z( TYPE* g_deriv, TYPE *g_field, 
									const int nx, const int ny, const int nz,
									const int nx_pad, const int ny_pad, const int nz_pad )
{
	int ix = blockIdx.x * tile_dimx + threadIdx.x;
	int iy = blockIdx.y * tile_dimy + threadIdx.y;
	int stride = nx_pad * ny_pad;

	int idx_out = iy * nx_pad + ix;
	int idx_in  = idx_out - radius * stride;

	TYPE queue[2*radius+1];

	#pragma unroll
	for( int i=1; i<2*radius+1; i++ )
	{
		queue[i] = __ldg( &g_field[idx_in] );
		idx_in += stride;
	}

	OUTER_UNROLL
	for( int iz=0; iz<nz; iz++ )
	{
		advance( queue, 2*radius+1 );
		queue[2*radius] = __ldg( &g_field[idx_in] );

		TYPE deriv = c_first[0] * queue[radius];
		#pragma unroll
		for( int i=1; i<=radius; i++ )
			deriv += c_first[i] * ( queue[radius+i] - queue[radius-i] );

		g_deriv[idx_out] = deriv;

		idx_in  += stride;
		idx_out += stride;
	}
}
#endif


#ifdef TRACE_DIFFLUX_UX
template<int tile_dimx, int tile_dimy, int radius>
__global__ void D_momentum_x( TYPE* g_difflux, TYPE *g_vx, TYPE *g_vy_y, TYPE *g_vz_z, 
							  const int nx, const int ny, const int nz,
							  const int nx_pad, const int ny_pad, const int nz_pad )
{
	TYPE queue_vx[2*radius+1];
	
	int ix = blockIdx.x * tile_dimx + threadIdx.x;
	int iy = blockIdx.y * tile_dimy + threadIdx.y;
	int stride = nx_pad * ny_pad;

	int idx_out = iy * nx_pad + ix;
	int idx_in  = idx_out - radius * stride;

	#pragma unroll
	for( int i=1; i<2*radius+1; i++ )
	{
		queue_vx[i] = __ldg( &g_vx[idx_in] );
		idx_in += stride;
	}

	OUTER_UNROLL
	for( int iz=0; iz<nz; iz++ )
	{
		advance( queue_vx, 2*radius+1 );
		queue_vx[2*radius] = __ldg( &g_vx[idx_in] );

		TYPE vx_xx, vx_yy, vx_zz, vy_xy, vz_xz;

		vx_xx = vx_yy = vx_zz = c_second[0] * queue_vx[radius];
		vy_xy = c_first[0] * __ldg( &g_vy_y[idx_out] ); // probably can be replaced with 0
		vz_xz = c_first[0] * __ldg( &g_vz_z[idx_out] ); // probably can be replaced with 0

		#pragma unroll
		for( int i=1; i<=radius; i++ )
		{
			vx_xx += c_second[i] * ( __ldg( &g_vx[idx_out+i] )        + __ldg( &g_vx[idx_out-i] ) );
			vx_yy += c_second[i] * ( __ldg( &g_vx[idx_out+i*nx_pad] ) + __ldg( &g_vx[idx_out-i*nx_pad] ) );
			vx_zz += c_second[i] * ( queue_vx[radius+i]               + queue_vx[radius-i]  );

			vy_xy += c_first[i] * ( __ldg( &g_vy_y[idx_out+i] )       - __ldg( &g_vy_y[idx_out-i] ) );
			vz_xz += c_first[i] * ( __ldg( &g_vz_z[idx_out+i] )       - __ldg( &g_vz_z[idx_out-i] ) );
		}

		g_difflux[idx_out] = FOUR_THIRDS * vx_xx + vx_yy + vx_zz + ONE_THIRD * ( vy_xy + vz_xz );

		idx_in  += stride;
		idx_out += stride;
	}
}
#endif


#ifdef TRACE_DIFFLUX_UY

// difflux for momentum in y
//   march in the y dimension
//   maintain 3 register-queues and 1 smem array
//
template<int tile_dimx, int tile_dimy, int radius>
__global__ void D_momentum_y_Q( TYPE* g_difflux, TYPE *g_vy, TYPE *g_vx_x, TYPE *g_vz_z, 
							  const int nx, const int ny, const int nz,
							  const int nx_pad, const int ny_pad, const int nz_pad )
{
	TYPE queue_vy[2*radius+1], queue_vx_x[2*radius+1], queue_vz_z[2*radius+1];

	int ix = blockIdx.x * tile_dimx + threadIdx.x;
	int iz = blockIdx.y * tile_dimy + threadIdx.y;
	int stride = nx_pad;

	int idx_out = iz * (nx_pad*ny_pad) + ix;
	int idx_in  = idx_out - radius * stride;

	#pragma unroll
	for( int i=1; i<2*radius+1; i++ )
	{
		queue_vy[i]   = __ldg( &g_vy[idx_in] );
		queue_vx_x[i] = __ldg( &g_vx_x[idx_in] );
		queue_vz_z[i] = __ldg( &g_vz_z[idx_in] );
		idx_in += stride;
	}

	OUTER_UNROLL
	for( int iy=0; iy<ny; iy++ )
	{
		advance( queue_vy, 2*radius+1 );
		queue_vy[2*radius] = __ldg( &g_vy[idx_in] );
		advance( queue_vx_x, 2*radius+1 );
		queue_vx_x[2*radius] = __ldg( &g_vx_x[idx_in] );
		advance( queue_vz_z, 2*radius+1 );
		queue_vz_z[2*radius] = __ldg( &g_vz_z[idx_in] );


		TYPE vy_xx, vy_yy, vy_zz, vx_xy, vz_yz;
		
		vy_xx = vy_yy = vy_zz = c_second[0] * queue_vy[radius];
		vx_xy = c_first[0] * queue_vx_x[radius];
		vz_yz = c_first[0] * queue_vz_z[radius];

		#pragma unroll
		for( int i=1; i<=radius; i++ )
		{
			vy_xx += c_second[i] * ( __ldg( &g_vy[idx_out+i] )               + __ldg( &g_vy[idx_out-i] ) );
			vy_yy += c_second[i] * ( __ldg( &g_vy[idx_out+i*nx_pad*ny_pad] ) + __ldg( &g_vy[idx_out-i*nx_pad*ny_pad] ) );
			vy_zz += c_second[i] * ( queue_vy[radius+i]                      + queue_vy[radius-i] );

			vx_xy += c_first[i] * ( queue_vx_x[radius+i] - queue_vx_x[radius-i] );
			vz_yz += c_first[i] * ( queue_vz_z[radius+i] - queue_vz_z[radius-i] );
		}

		g_difflux[idx_out] = vy_xx + FOUR_THIRDS * vy_yy + vy_zz + ONE_THIRD * ( vx_xy + vz_yz );

		idx_in  += stride;
		idx_out += stride;
	}
}
#endif


#ifdef TRACE_DIFFLUX_UZ

// difflux for momentum in z
//   march in the z dimension
//   maintain 3 register-queues and 1 smem array
//
template<int tile_dimx, int tile_dimy, int radius>
__global__ void D_momentum_z_Q( TYPE* g_difflux, TYPE *g_vz, TYPE *g_vx_x, TYPE *g_vy_y, 
							  const int nx, const int ny, const int nz,
							  const int nx_pad, const int ny_pad, const int nz_pad )
{
	TYPE queue_vz[2*radius+1];
	TYPE queue_vx_x[2*radius+1];
	TYPE queue_vy_y[2*radius+1];
	
	int ix = blockIdx.x * tile_dimx + threadIdx.x;
	int iy = blockIdx.y * tile_dimy + threadIdx.y;
	int stride = nx_pad * ny_pad;

	int idx_out = iy * nx_pad + ix;
	int idx_in  = idx_out - radius * stride;

	#pragma unroll
	for( int i=1; i<2*radius+1; i++ )
	{
		queue_vz[i]   = __ldg( &g_vz[idx_in] );
		queue_vx_x[i] = __ldg( &g_vx_x[idx_in] );
		queue_vy_y[i] = __ldg( &g_vy_y[idx_in] );
		idx_in += stride;
	}

	OUTER_UNROLL
	for( int iz=0; iz<nz; iz++ )
	{
		advance( queue_vz, 2*radius+1 );
		queue_vz[2*radius] = __ldg( &g_vz[idx_in] );

		advance( queue_vx_x, 2*radius+1 );
		queue_vx_x[2*radius] = __ldg( &g_vx_x[idx_in] );
		
		advance( queue_vy_y, 2*radius+1 );
		queue_vy_y[2*radius] = __ldg( &g_vy_y[idx_in] );

		TYPE vz_xx, vz_yy, vz_zz, vx_xz, vy_yz;
		
		vz_xx = vz_yy = vz_zz = c_second[0] * queue_vz[radius];
		vx_xz = c_first[0] * queue_vx_x[radius];
		vy_yz = c_first[0] * queue_vy_y[radius];

		#pragma unroll
		for( int i=1; i<=radius; i++ )
		{
			vz_xx += c_second[i] * ( __ldg( &g_vz[idx_out+i] )        + __ldg( &g_vz[idx_out-i] ) );
			vz_yy += c_second[i] * ( queue_vz[radius+i]               + queue_vz[radius-i] );
			vz_zz += c_second[i] * ( __ldg( &g_vz[idx_out+i*nx_pad] ) + __ldg( &g_vz[idx_out-i*nx_pad] ) );

			vx_xz += c_first[i] * ( queue_vx_x[radius+i] - queue_vx_x[radius-i] );
			vy_yz += c_first[i] * ( queue_vy_y[radius+i] - queue_vy_y[radius-i] );
		}

		g_difflux[idx_out] = vz_xx + vz_yy + FOUR_THIRDS * vz_zz + ONE_THIRD * ( vx_xz + vy_yz );

		idx_in  += stride;
		idx_out += stride;
	}
}
#endif


#ifdef TRACE_DIFFLUX_E

// difflux of energy
//   marches in z
//   reads 3 velocities and 3 components of momentum-difflux from gmem
//   computes 3 temperature derivatives and 9 velocity derivatives on the fly
//
template<int tile_dimx, int tile_dimy, int radius>
__global__ void D_energy_4q( 
						  TYPE *g_difflux, 
	                      const TYPE alam, const TYPE eta,
						  TYPE *g_T,
	                      TYPE *g_vx, TYPE *g_vy, TYPE *g_vz,
						  TYPE *g_difflux_x, TYPE *g_difflux_y, TYPE *g_difflux_z,
						  const int nx, const int ny, const int nz,
						  const int nx_pad, const int ny_pad, const int nz_pad )
{
	TYPE queue_T[2*radius+1];
	TYPE queue_vx[2*radius+1], queue_vy[2*radius+1], queue_vz[2*radius+1];

	int ix = blockIdx.x * tile_dimx + threadIdx.x;
	int iy = blockIdx.y * tile_dimy + threadIdx.y;
	int stride = nx_pad*ny_pad;

	int idx_out = iy * nx_pad + ix;
	int idx_in  = idx_out - radius * stride;
	
	#pragma unroll
	for( int i=1; i<2*radius+1; i++ )
	{
		queue_T[i]  = __ldg( &g_T[idx_in] );
		queue_vx[i] = __ldg( &g_vx[idx_in] );
		queue_vy[i] = __ldg( &g_vy[idx_in] );
		queue_vz[i] = __ldg( &g_vz[idx_in] );

		idx_in += stride;
	}

	OUTER_UNROLL
	for( int iz=0; iz<nz; iz++ )
	{
		advance( queue_T, 2*radius+1 );
		queue_T[2*radius] = __ldg( &g_T[idx_in] );
		advance( queue_vx, 2*radius+1 );
		queue_vx[2*radius] = __ldg( &g_vx[idx_in] );
		advance( queue_vy, 2*radius+1 );
		queue_vy[2*radius] = __ldg( &g_vy[idx_in] );
		advance( queue_vz, 2*radius+1 );
		queue_vz[2*radius] = __ldg( &g_vz[idx_in] );

		TYPE vx_x=0, vx_y=0, vx_z=0;
		TYPE vy_x=0, vy_y=0, vy_z=0;
		TYPE vz_x=0, vz_y=0, vz_z=0;
		
		TYPE laplacian_T = 3 * c_second[0] * queue_T[radius];

		#pragma unroll
		for( int i=1; i<=radius; i++ )
		{
			laplacian_T += c_second[i] * ( queue_T[radius+i]               + queue_T[radius-i] +
				                           __ldg( &g_T[idx_out+i] )        + __ldg( &g_T[idx_out-i] ) +
										   __ldg( &g_T[idx_out+i*nx_pad] ) + __ldg( &g_T[idx_out-i*nx_pad] ) );

			vx_x += c_first[i] * ( __ldg( &g_vx[idx_out+i] )               - __ldg( &g_vx[idx_out-i] ) );
			vx_y += c_first[i] * ( __ldg( &g_vx[idx_out+i*nx_pad] )        - __ldg( &g_vx[idx_out-i*nx_pad] ) );
			vx_z += c_first[i] * ( queue_vx[radius+i]                      - queue_vx[radius-i] );
			
			vy_x += c_first[i] * ( __ldg( &g_vy[idx_out+i] )               - __ldg( &g_vy[idx_out-i] ) );
			vy_y += c_first[i] * ( __ldg( &g_vy[idx_out+i*nx_pad] )        - __ldg( &g_vy[idx_out-i*nx_pad] ) );
			vy_z += c_first[i] * ( queue_vy[radius+i]                      - queue_vy[radius-i] );

			vz_x += c_first[i] * ( __ldg( &g_vz[idx_out+i] )               - __ldg( &g_vz[idx_out-i] ) );
			vz_y += c_first[i] * ( __ldg( &g_vz[idx_out+i*nx_pad] )        - __ldg( &g_vz[idx_out-i*nx_pad] ) );
			vz_z += c_first[i] * ( queue_vz[radius+i]                      - queue_vz[radius-i] );
		}

		TYPE divu  = TWO_THIRDS * ( vx_x + vy_y + vz_z );
		TYPE tauxx = 2 * vx_x - divu;
		TYPE tauyy = 2 * vy_y - divu;
		TYPE tauzz = 2 * vz_z - divu;
		TYPE tauxy = vx_y + vy_x;
		TYPE tauxz = vx_z + vz_x;
		TYPE tauyz = vy_z + vz_y;

		TYPE mechwork = tauxx*vx_x + tauyy*vy_y + tauzz*vz_z + 
			            tauxy*tauxy + tauxz*tauxz + tauyz*tauyz;

		mechwork = eta * mechwork + 
			       __ldg( &g_difflux_x[idx_out] ) * __ldg( &g_vx[idx_out] ) + 
				   __ldg( &g_difflux_y[idx_out] ) * __ldg( &g_vy[idx_out] ) + 
				   __ldg( &g_difflux_z[idx_out] ) * __ldg( &g_vz[idx_out] );

		g_difflux[idx_out] = alam * laplacian_T + mechwork;

		idx_in  += stride;
		idx_out += stride;
	}
}
#endif


#ifdef TRACE_UPDATE_E

// 2 queues (redundant reads of rE)
//   all LDGs use index arithmetic and array notation
//
template<int tile_dimx, int tile_dimy, int radius>
__global__ void update_rE( 
	              TYPE *g_rE, 
				  const TYPE *g_p, 
				  const TYPE *g_vx, const TYPE *g_vy, const TYPE *g_vz,
				  const TYPE *g_difflux_E,
				  const TYPE delta_t,
	              const int nx, const int ny, const int nz,
                  const int nx_pad, const int ny_pad, const int nz_pad )
{
	int ix = blockIdx.x * tile_dimx + threadIdx.x;
	int iy = blockIdx.y * tile_dimy + threadIdx.y;

	int slice_stride = nx_pad * ny_pad;

	int idx_out = iy * nx_pad + ix;
	int idx_in  = idx_out - radius * slice_stride;

	const int diameter = 2*radius + 1;

	TYPE queue_vz[2*radius+1];
	TYPE queue_rE_p[2*radius+1];

	#pragma unroll
	for( int i=1; i<diameter; i++ )
	{
		queue_rE_p[i] = __ldg( &g_rE[idx_in] ) + __ldg( &g_p[idx_in] );
		queue_vz[i]   = __ldg( &g_vz[idx_in] );
		idx_in += slice_stride;
	}

	OUTER_UNROLL
	for( int iz=0; iz<nz; iz++ )
	{
		advance( queue_rE_p, diameter );
		queue_rE_p[diameter-1] = __ldg( &g_rE[idx_in]) + __ldg( &g_p[idx_in] );
		advance( queue_vz, diameter );
		queue_vz[diameter-1] = __ldg( &g_vz[idx_in] );

		TYPE rEp_vx_x = 0;
		TYPE rEp_vy_y = 0;
		TYPE rEp_vz_z = 0;
				
		#pragma unroll
		for( int i=1; i<=radius; i++ )
		{
			rEp_vx_x += c_first[i] * ( __ldg( &g_vx[idx_out+i] ) * ( __ldg( &g_rE[idx_out+i] ) + __ldg( &g_p[idx_out+i] ) ) - 
				                       __ldg( &g_vx[idx_out-i] ) * ( __ldg( &g_rE[idx_out-i] ) + __ldg( &g_p[idx_out-i] ) ) );
			rEp_vy_y += c_first[i] * ( __ldg( &g_vx[idx_out+i*nx_pad] ) * ( __ldg( &g_rE[idx_out+i*nx_pad] ) + __ldg( &g_p[idx_out+i*nx_pad] ) ) - 
				                       __ldg( &g_vx[idx_out-i*nx_pad] ) * ( __ldg( &g_rE[idx_out-i*nx_pad] ) + __ldg( &g_p[idx_out-i*nx_pad] ) ) );
			rEp_vz_z += c_first[i] * ( queue_rE_p[radius+i]*queue_vz[radius+i] - queue_rE_p[radius-i]*queue_vz[radius-i] );
		}

		g_rE[idx_out] = __ldg( &g_rE[idx_out] ) + delta_t * ( __ldg( &g_difflux_E[idx_out] ) - ( rEp_vx_x + rEp_vy_y + rEp_vz_z ) );

		idx_in  += slice_stride;
		idx_out += slice_stride;
	}
}
#endif


#ifdef TRACE_UPDATE_U

// updates all 3 momentum components
// computes all 3 hypterm components, reads momenta and density
// 4 shared memory arrays
//
template<int tile_dimx, int tile_dimy, int radius>
__global__ void update_xyz_m4s( 
                  TYPE *g_ux, TYPE *g_uy, TYPE *g_uz,
				  const TYPE *g_rho, const TYPE *g_p,
				  const TYPE *g_difflux_x, const TYPE *g_difflux_y, const TYPE *g_difflux_z,
				  const TYPE delta_t,
	              const int nx, const int ny, const int nz,
                  const int nx_pad, const int ny_pad, const int nz_pad )
{
	TYPE l_ux[2*radius+1], l_uy[2*radius+1], l_uz[2*radius+1], l_rho[2*radius+1], l_p[2*radius+1];

	int ix = blockIdx.x * tile_dimx + threadIdx.x;
	int iy = blockIdx.y * tile_dimy + threadIdx.y;

	int slice_stride = nx_pad * ny_pad;

	int idx_out = iy * nx_pad + ix;
	int idx_in  = idx_out - radius * slice_stride;

	const int diameter = 2*radius + 1;

	#pragma unroll
	for( int i=1; i<diameter; i++ )
	{
		l_ux[i]  = __ldg( &g_ux[idx_in] );
		l_uy[i]  = __ldg( &g_uy[idx_in] );
		l_uz[i]  = __ldg( &g_uz[idx_in] );
		l_rho[i] = __ldg( &g_rho[idx_in] );
		l_p[i]   = __ldg( &g_p[idx_in] );

		idx_in += slice_stride;
	}

	OUTER_UNROLL
	for( int iz=0; iz<nz; iz++ )
	{		
		advance( l_ux, diameter );
		l_ux[diameter-1] = __ldg( &g_ux[idx_in] );
		advance( l_uy, diameter );
		l_uy[diameter-1] = __ldg( &g_uy[idx_in] );
		advance( l_uz, diameter );
		l_uz[diameter-1] = __ldg( &g_uz[idx_in] );
		advance( l_rho, diameter );
		l_rho[diameter-1] = __ldg( &g_rho[idx_in] );
		advance( l_p, diameter );
		l_p[diameter-1] = __ldg( &g_p[idx_in] );
		
		TYPE hypterm_x=0.f, hypterm_y=0.f, hypterm_z=0.f;

		#pragma unroll
		for( int i=1; i<=radius; i++ )
		{
			hypterm_x += c_first[i] * (	
				( __ldg( &g_rho[idx_out+i] ) *__ldg( &g_ux[idx_out+i] ) * __ldg( &g_ux[idx_out+i] )                       - __ldg( &g_rho[idx_out-i] )*__ldg( &g_ux[idx_out-i] )*__ldg( &g_ux[idx_out-i] ) ) +                       // x-deriv of rho-ux-ux product
				( __ldg( &g_rho[idx_out+i*nx_pad] ) * __ldg( &g_ux[idx_out+i*nx_pad] ) * __ldg( &g_uy[idx_out+i*nx_pad] ) - __ldg( &g_rho[idx_out-i*nx_pad] )*__ldg( &g_ux[idx_out-i*nx_pad] )*__ldg( &g_uy[idx_out-i*nx_pad] ) ) +  // y-deriv of rho-ux-uy product
				( l_rho[radius+i] * l_ux[radius+i] * l_uz[radius+i]                                                       - l_rho[radius-i]*l_ux[radius-i]*l_uz[radius-i] ) +                                                        // z-deriv of rho-ux-uz product
				( __ldg( &g_p[idx_out+i] )                                                                                - __ldg( &g_p[idx_out-i] ) )                                                                               // x-deriv of pressure
			);

			hypterm_y += c_first[i] * (	
				( __ldg( &g_rho[idx_out+i] ) * __ldg( &g_ux[idx_out+i] ) * __ldg( &g_uy[idx_out+i] )                      - __ldg( &g_rho[idx_out-i] ) *__ldg( &g_ux[idx_out-i] ) * __ldg( &g_uy[idx_out-i] ) ) +                       // x-deriv of rho-ux-uy product
				( __ldg( &g_rho[idx_out+i*nx_pad] ) * __ldg( &g_uy[idx_out+i*nx_pad] ) * __ldg( &g_uy[idx_out+i*nx_pad] ) - __ldg( &g_rho[idx_out-i*nx_pad] ) *__ldg( &g_uy[idx_out-i*nx_pad] ) * __ldg( &g_uy[idx_out-i*nx_pad] ) ) +  // y-deriv of rho-uy-uy product
				( l_rho[radius+i]*l_uy[radius+i]*l_uz[radius+i]                                                           - l_rho[radius-i]*l_uy[radius-i]*l_uz[radius-i] ) +                                                           // z-deriv of rho-uy-uz product
				( __ldg( &g_p[idx_out+i*nx_pad] )                                                                         - __ldg( &g_p[idx_out-i*nx_pad] ) )                                                                           // y-deriv of pressure
			);

			hypterm_z += c_first[i] * (	
				( __ldg( &g_rho[idx_out+i] ) * __ldg( &g_ux[idx_out+i] ) * __ldg( &g_uz[idx_out+i] )                      - __ldg( &g_rho[idx_out-i] ) * __ldg( &g_ux[idx_out-i] ) * __ldg( &g_uz[idx_out-i] ) ) +                       // x-deriv of rho-ux-uz product
				( __ldg( &g_rho[idx_out+i*nx_pad] ) * __ldg( &g_uy[idx_out+i*nx_pad] ) * __ldg( &g_uz[idx_out+i*nx_pad] ) - __ldg( &g_rho[idx_out-i*nx_pad] ) * __ldg( &g_uy[idx_out-i*nx_pad] ) * __ldg( &g_uz[idx_out-i*nx_pad] ) ) +  // y-deriv of rho-uy-uz product
				( l_rho[radius+i]*l_uz[radius+i]*l_uz[radius+i]                                                           - l_rho[radius-i]*l_uz[radius-i]*l_uz[radius-i] ) +                                                            // z-deriv of rho-uz-uz product
				( l_p[radius+i]                                                                                           - l_p[radius-i] )                                                                                              // z-deriv of pressure
			);
		}

		g_ux[idx_out] = l_ux[radius] + __ldg( &g_difflux_x[idx_out] ) - delta_t * hypterm_x;
		g_uy[idx_out] = l_uy[radius] + __ldg( &g_difflux_y[idx_out] ) - delta_t * hypterm_y;
		g_uz[idx_out] = l_uz[radius] + __ldg( &g_difflux_z[idx_out] ) - delta_t * hypterm_z;

		idx_in  += slice_stride;
		idx_out += slice_stride;
	}
}
#endif


#ifdef TRACE_UPDATE_U_ALT

// updates all 3 momentum components
//   computes all 3 hypterm components, reads momenta and density
//   4 shared memory arrays
//   computes all derivatives first, then applies product rule
//

template<int tile_dimx, int tile_dimy, int radius>
//__launch_bounds__(256,2)
__global__ void update_xyz_m4s_p( 
                  TYPE *g_ux, TYPE *g_uy, TYPE *g_uz,
				  const TYPE *g_rho, const TYPE *g_p,
				  const TYPE *g_difflux_x, const TYPE *g_difflux_y, const TYPE *g_difflux_z,
				  const TYPE delta_t,
	              const int nx, const int ny, const int nz,
                  const int nx_pad, const int ny_pad, const int nz_pad )
{
	TYPE l_ux[2*radius+1], l_uy[2*radius+1], l_uz[2*radius+1], l_rho[2*radius+1], l_p[2*radius+1];

	int ix = blockIdx.x * tile_dimx + threadIdx.x;
	int iy = blockIdx.y * tile_dimy + threadIdx.y;

	int slice_stride = nx_pad * ny_pad;

	int idx_out = iy * nx_pad + ix;
	int idx_in  = idx_out - radius * slice_stride;

	const int diameter = 2*radius + 1;

	#pragma unroll
	for( int i=1; i<diameter; i++ )
	{
		l_ux[i]  = __ldg( &g_ux[idx_in] );
		l_uy[i]  = __ldg( &g_uy[idx_in] );
		l_uz[i]  = __ldg( &g_uz[idx_in] );
		l_rho[i] = __ldg( &g_rho[idx_in] );
		l_p[i]   = __ldg( &g_p[idx_in] );

		idx_in += slice_stride;
	}

	OUTER_UNROLL
	for( int iz=0; iz<nz; iz++ )
	{
		TYPE p_x=0, p_y=0, p_z=0;

		advance( l_ux, diameter );
		l_ux[diameter-1] = __ldg( &g_ux[idx_in] );
		advance( l_uy, diameter );
		l_uy[diameter-1] = __ldg( &g_uy[idx_in] );
		advance( l_uz, diameter );
		l_uz[diameter-1] = __ldg( &g_uz[idx_in] );
		advance( l_rho, diameter );
		l_rho[diameter-1] = __ldg( &g_rho[idx_in] );
		advance( l_p, diameter );
		l_p[diameter-1] = __ldg( &g_p[idx_in] );


		////////////////////////////////////
		// compute the x component
		//
		TYPE rho_x=0, rho_y=0, rho_z=0;
		TYPE ux_x=0, ux_y=0, ux_z=0;
		TYPE uy_x=0, uy_y=0, uy_z=0;
		TYPE uz_x=0, uz_y=0, uz_z=0;
		
		#pragma unroll
		for( int i=1; i<=radius; i++ )
		{
			rho_x += c_first[i] * (	__ldg( &g_rho[idx_out+i] ) - __ldg( &g_rho[idx_out-i] ) );
			ux_x  += c_first[i] * (	__ldg( &g_ux[idx_out+i]  ) - __ldg( &g_ux[idx_out-i]  ) );
			uy_x  += c_first[i] * (	__ldg( &g_uy[idx_out+i]  ) - __ldg( &g_uy[idx_out-i]  ) );
			uz_x  += c_first[i] * (	__ldg( &g_uz[idx_out+i]  ) - __ldg( &g_uz[idx_out-i]  ) );
			p_x   += c_first[i] * ( __ldg( &g_p[idx_out+i]   ) - __ldg( &g_p[idx_out-i] ) );

			rho_y += c_first[i] * (	__ldg( &g_rho[idx_out+i*nx_pad] ) - __ldg( &g_rho[idx_out-i*nx_pad] ) );
			ux_y  += c_first[i] * (	__ldg( &g_ux[idx_out+i*nx_pad]  ) - __ldg( &g_ux[idx_out-i*nx_pad]  ) );
			uy_y  += c_first[i] * (	__ldg( &g_uy[idx_out+i*nx_pad]  ) - __ldg( &g_uy[idx_out-i*nx_pad]  ) );
			uz_y  += c_first[i] * (	__ldg( &g_uz[idx_out+i*nx_pad]  ) - __ldg( &g_uz[idx_out-i*nx_pad]  ) );
			p_y   += c_first[i] * ( __ldg( &g_p[idx_out+i*nx_pad] )   - __ldg( &g_p[idx_out-i*nx_pad] ) );

			rho_z += c_first[i] * (	l_rho[radius+i] - l_rho[radius-i] );
			ux_z  += c_first[i] * (	l_ux[radius+i]  - l_ux[radius-i]  );
			uy_z  += c_first[i] * (	l_uy[radius+i]  - l_uy[radius-i]  );
			uz_z  += c_first[i] * (	l_uz[radius+i]  - l_uz[radius-i]  );
			p_z   += c_first[i] * ( l_p[radius+i]   - l_p[radius-i] );
		}

		TYPE hypterm_x =   ( rho_x*l_ux[radius]*l_ux[radius] + 2*l_rho[radius]*ux_x*l_ux[radius] ) 
			             + ( rho_y*l_ux[radius]*l_uy[radius] + l_rho[radius]*ux_y*l_uy[radius] + l_rho[radius]*l_ux[radius]*uy_y ) 
						 + ( rho_z*l_ux[radius]*l_uz[radius] + l_rho[radius]*ux_z*l_uz[radius] + l_rho[radius]*l_ux[radius]*uz_z ) 
			             + p_x;
		//__threadfence_block();
		TYPE hypterm_y =   ( rho_x*l_ux[radius]*l_uy[radius] + l_rho[radius]*ux_x*l_uy[radius] + l_rho[radius]*l_ux[radius]*uy_x ) 
			             + ( rho_y*l_uy[radius]*l_uy[radius] + 2*l_rho[radius]*uy_y*l_uy[radius] ) 
						 + ( rho_z*l_uy[radius]*l_uz[radius] + l_rho[radius]*uy_z*l_uz[radius] + l_rho[radius]*l_uy[radius]*uz_z ) 
			             + p_y;
		//__threadfence_block();
		TYPE hypterm_z =   ( rho_x*l_ux[radius]*l_uz[radius] + l_rho[radius]*ux_x*l_uz[radius] + l_rho[radius]*l_ux[radius]*uz_x ) 
			             + ( rho_y*l_uy[radius]*l_uz[radius] + l_rho[radius]*uy_y*l_uz[radius] + l_rho[radius]*l_uy[radius]*uz_y )
						 + ( rho_z*l_uz[radius]*l_uz[radius] + 2*l_rho[radius]*uz_z*l_uz[radius] ) 
			             + p_z;

		g_ux[idx_out] = l_ux[radius] + __ldg( &g_difflux_x[idx_out] ) - delta_t * hypterm_x;
		g_uy[idx_out] = l_uy[radius] + __ldg( &g_difflux_y[idx_out] ) - delta_t * hypterm_y;
		g_uz[idx_out] = l_uz[radius] + __ldg( &g_difflux_z[idx_out] ) - delta_t * hypterm_z;

		__syncthreads();
		
		idx_in  += slice_stride;
		idx_out += slice_stride;
	}
}
#endif

#define NUM_VOLS 8

int main( int argc, char *argv[] )
{
	int nx = 8*32;
	int ny = 8*32;
	int nz = 8*32;
	int num_iterations = 1;
	int gpu_id = 0;

	/////////////////////////////////////////
	// process command-line arguments
	//
	if( argc >= 4 )
	{
		nx = atoi( argv[1] );
		ny = atoi( argv[2] );
		nz = atoi( argv[3] );
	}
	if( argc >= 5 )
		num_iterations = atoi( argv[4] );
	if( argc >= 6 )
		gpu_id = atoi( argv[5] );

	TYPE delta_t = 0.01;

	/////////////////////////////////////////
	// create GPu context
	//
	cudaError_t error = cudaSuccess;
	
	error = cudaSetDevice( gpu_id );
	process_error( error, "set device" );
	cudaDeviceProp prop;
	error = cudaGetDeviceProperties( &prop, gpu_id );
	process_error( error, "get device properties" );
	printf("%s\n", prop.name );
	error = cudaFree( 0 );
	process_error( error, "create GPU context" );
	
	printf( "%d %d %d  %d\n", nx, ny, nz, num_iterations );

	
	/////////////////////////////////////////
	// allocate memory on GPU 
	//

	TYPE *d_volumes[NUM_VOLS];

	const int radius = 4;
	const int cache_line_size_type = CACHE_LINE_SIZE / sizeof(TYPE);

	size_t nx_pad = nx + 2 * radius;
	size_t ny_pad = ny + 2 * radius;
	size_t nz_pad = nz + 2 * radius;

	nx_pad += cache_line_size_type - ( nx_pad % cache_line_size_type ); // make each row a multiple of cache-line size
	size_t lead_pad = cache_line_size_type - radius;
	
	size_t num_bytes_padded = ( lead_pad + nx_pad * ny_pad * nz_pad ) * sizeof( TYPE );
	size_t num_bytes        = nx * ny * nz * sizeof( TYPE );

	int padding_to_first_output_cell = lead_pad + radius*nx_pad*ny_pad + radius*nx_pad + radius;

    for( int i=0; i<NUM_VOLS; i++ )
	{
		error = cudaMalloc( &d_volumes[i], num_bytes_padded );
		char message[20];
		sprintf( message, "allocate vol %d", i );
		process_error( error, message );

		d_volumes[i] += padding_to_first_output_cell;
	}
	printf( "%dx%dx%d %8.3f GB\n", nx_pad, ny_pad, nz_pad, (((float)NUM_VOLS)*num_bytes_padded)/(1024.f*1024.f*1024.f) );

//	error = cudaDeviceSetSharedMemConfig( cudaSharedMemBankSizeEightByte );
//	process_error( error, "set SMEM addressing mode to 8-byte" );
/*
	cudaEvent_t start, stop;
	error = cudaEventCreate( &start );
	process_error( error, "create start event" );
	error = cudaEventCreate( &stop );
	process_error( error, "create stop event" );
*/
	float elapsed_time_ms=0.f, throughput_mcells=0.f;

#ifdef TRACE_UPDATE_RHO
	{	// update rho, march in z, 4 volumes

		TYPE *d_rho        = d_volumes[0];
		TYPE *d_momentum_x = d_volumes[1];
		TYPE *d_momentum_y = d_volumes[2];
		TYPE *d_momentum_z = d_volumes[3];

		dim3 block( TILE_DIMX, TILE_DIMY );
		dim3 grid( nx/block.x, ny/block.y );

//		cudaEventRecord( start, 0 );
		for( int i=0; i<num_iterations; i++ )
		{
			update_rho_single_pass<TILE_DIMX,TILE_DIMY,RADIUS><<<grid,block>>>( 
				d_rho, d_momentum_x, d_momentum_y, d_momentum_z, 
				delta_t,
				nx, ny, nz,
				nx_pad, ny_pad, nz_pad );
		}
//		cudaEventRecord( stop, 0 );

		cudaDeviceSynchronize();
		error = cudaGetLastError();
		process_error( error, "rho kernel" );

/*		error = cudaEventElapsedTime( &elapsed_time_ms, start, stop );
		process_error( error, "get kernel time" );
	
		elapsed_time_ms /= num_iterations;
		throughput_mcells = 1e-3f * nx*ny*nz / elapsed_time_ms;

		printf( "%15s:  %7.2f ms, %7.1f MC/s\n", "rho", elapsed_time_ms, throughput_mcells );
*/
	}
#endif

#ifdef TRACE_DX
	{	// derivative in x, march in z, 2 volumes

		TYPE *d_vx_x = d_volumes[0];
		TYPE *d_vx   = d_volumes[1];

		dim3 block( TILE_DIMX, TILE_DIMY );
		dim3 grid( nx/block.x, ny/block.y );

//		cudaEventRecord( start, 0 );
		for( int i=0; i<num_iterations; i++ )
		{
			first_derivative_x<TILE_DIMX,TILE_DIMY,RADIUS><<<grid,block>>>( 
				d_vx_x, d_vx,
				nx, ny, nz,
				nx_pad, ny_pad, nz_pad );
		}
//		cudaEventRecord( stop, 0 );

		cudaDeviceSynchronize();
		error = cudaGetLastError();
		process_error( error, "x kernel" );

/*		error = cudaEventElapsedTime( &elapsed_time_ms, start, stop );
		process_error( error, "get kernel time" );
	
		elapsed_time_ms /= num_iterations;
		throughput_mcells = 1e-3f * nx*ny*nz / elapsed_time_ms;

		printf( "%15s:  %7.2f ms, %7.1f MC/s\n", "vx_x", elapsed_time_ms, throughput_mcells );
*/
	}
#endif

#ifdef TRACE_DZ
	{	// derivative in z, march in z, 2 volumes

		TYPE *d_vz_z = d_volumes[0];
		TYPE *d_vz  = d_volumes[1];

		dim3 block( TILE_DIMX, TILE_DIMY );
		dim3 grid( nx/block.x, ny/block.y );

//		cudaEventRecord( start, 0 );
		for( int i=0; i<num_iterations; i++ )
		{
			first_derivative_z<TILE_DIMX,TILE_DIMY,RADIUS><<<grid,block>>>( 
				d_vz_z, d_vz,
				nx, ny, nz,
				nx_pad, ny_pad, nz_pad );
		}
//		cudaEventRecord( stop, 0 );

		cudaDeviceSynchronize();
		error = cudaGetLastError();
		process_error( error, "z kernel" );

/*		error = cudaEventElapsedTime( &elapsed_time_ms, start, stop );
		process_error( error, "get kernel time" );
	
		elapsed_time_ms /= num_iterations;
		throughput_mcells = 1e-3f * nx*ny*nz / elapsed_time_ms;

		printf( "%15s:  %7.2f ms, %7.1f MC/s\n", "vz_z", elapsed_time_ms, throughput_mcells );
*/
	}
#endif

#ifdef TRACE_DIFFLUX_UX
	{	// difflux momentum-x, march in z, 4 volumes

		TYPE *d_difflux_x = d_volumes[0];
		TYPE *d_vx        = d_volumes[1];
		TYPE *d_vy_y      = d_volumes[2];
		TYPE *d_vz_z      = d_volumes[3];

		dim3 block( TILE_DIMX, TILE_DIMY );
		dim3 grid( nx/block.x, ny/block.y );

//		cudaEventRecord( start, 0 );
		for( int i=0; i<num_iterations; i++ )
		{
			D_momentum_x<TILE_DIMX,TILE_DIMY,RADIUS><<<grid,block>>>( 
				d_difflux_x, d_vx, d_vy_y, d_vz_z,
				nx, ny, nz,
				nx_pad, ny_pad, nz_pad );
		}
//		cudaEventRecord( stop, 0 );

		cudaDeviceSynchronize();
		error = cudaGetLastError();
		process_error( error, "difflux-x kernel" );

/*		error = cudaEventElapsedTime( &elapsed_time_ms, start, stop );
		process_error( error, "get kernel time" );
	
		elapsed_time_ms /= num_iterations;
		throughput_mcells = 1e-3f * nx*ny*nz / elapsed_time_ms;

		printf( "%15s:  %7.2f ms, %7.1f MC/s\n", "difflux_x", elapsed_time_ms, throughput_mcells );
*/
	}
#endif

#ifdef TRACE_DIFFLUX_UY
	{	// difflux momentum-y, march in y, 4 volumes

		TYPE *d_difflux_y = d_volumes[0];
		TYPE *d_vy        = d_volumes[1];
		TYPE *d_vx_x      = d_volumes[2];
		TYPE *d_vz_z      = d_volumes[3];

		dim3 block( TILE_DIMX, TILE_DIMY );
		dim3 grid( nx/block.x, nz/block.y );

//		cudaEventRecord( start, 0 );
		for( int i=0; i<num_iterations; i++ )
		{
			D_momentum_y_Q<TILE_DIMX,TILE_DIMY,RADIUS><<<grid,block>>>( 
				d_difflux_y, d_vy, d_vx_x, d_vz_z,
				nx, ny, nz,
				nx_pad, ny_pad, nz_pad );
		}
//		cudaEventRecord( stop, 0 );

		cudaDeviceSynchronize();
		error = cudaGetLastError();
		process_error( error, "difflux-yQ kernel" );

/*		error = cudaEventElapsedTime( &elapsed_time_ms, start, stop );
		process_error( error, "get kernel time" );
	
		elapsed_time_ms /= num_iterations;
		throughput_mcells = 1e-3f * nx*ny*nz / elapsed_time_ms;

		printf( "%15s:  %7.2f ms, %7.1f MC/s\n", "difflux_yQ", elapsed_time_ms, throughput_mcells );
*/
	}
#endif

#ifdef TRACE_DIFFLUX_UZ
	{	// difflux momentum-x, march in z, 4 volumes

		TYPE *d_difflux_z = d_volumes[0];
		TYPE *d_vz        = d_volumes[1];
		TYPE *d_vx_x      = d_volumes[2];
		TYPE *d_vy_y      = d_volumes[3];

		dim3 block( TILE_DIMX, TILE_DIMY );
		dim3 grid( nx/block.x, ny/block.y );

//		cudaEventRecord( start, 0 );
		for( int i=0; i<num_iterations; i++ )
		{
			D_momentum_z_Q<TILE_DIMX,TILE_DIMY,RADIUS><<<grid,block>>>( 
				d_difflux_z, d_vz, d_vx_x, d_vy_y,
				nx, ny, nz,
				nx_pad, ny_pad, nz_pad );
		}
//		cudaEventRecord( stop, 0 );

		cudaDeviceSynchronize();
		error = cudaGetLastError();
		process_error( error, "difflux-zQ kernel" );

/*		error = cudaEventElapsedTime( &elapsed_time_ms, start, stop );
		process_error( error, "get kernel time" );
	
		elapsed_time_ms /= num_iterations;
		throughput_mcells = 1e-3f * nx*ny*nz / elapsed_time_ms;

		printf( "%15s:  %7.2f ms, %7.1f MC/s\n", "difflux_zQ", elapsed_time_ms, throughput_mcells );
*/
	}
#endif

#ifdef TRACE_DIFFLUX_E
	{	// difflux energy, march in z, 8 volumes

		TYPE *d_difflux_E = d_volumes[0];
		TYPE *d_vx        = d_volumes[1];
		TYPE *d_vy        = d_volumes[2];
		TYPE *d_vz        = d_volumes[3];
		TYPE *d_difflux_x = d_volumes[4];
		TYPE *d_difflux_y = d_volumes[5];
		TYPE *d_difflux_z = d_volumes[6];
		TYPE *d_T         = d_volumes[7];

		dim3 block( TILE_DIMX, TILE_DIMY );
		dim3 grid( nx/block.x, ny/block.y );

		TYPE alam = 5.3;
		TYPE eta  = 7.1;

//		cudaEventRecord( start, 0 );
		for( int i=0; i<num_iterations; i++ )
		{
			D_energy_4q<TILE_DIMX,TILE_DIMY,RADIUS><<<grid,block>>>( 
				d_difflux_E, 
				alam, eta,
				d_T,
				d_vx, d_vy, d_vz,
				d_difflux_x, d_difflux_y, d_difflux_z,
				nx, ny, nz,
				nx_pad, ny_pad, nz_pad );
		}
//		cudaEventRecord( stop, 0 );

		cudaDeviceSynchronize();
		error = cudaGetLastError();
		process_error( error, "difflux-E_4q kernel" );

/*		error = cudaEventElapsedTime( &elapsed_time_ms, start, stop );
		process_error( error, "get kernel time" );
	
		elapsed_time_ms /= num_iterations;
		throughput_mcells = 1e-3f * nx*ny*nz / elapsed_time_ms;

		printf( "%15s:  %7.2f ms, %7.1f MC/s\n", "difflux_E_4q", elapsed_time_ms, throughput_mcells );
*/
	}
#endif

#ifdef TRACE_UPDATE_E
	{	// update energy, march in z, 6 volumes

		TYPE *d_rE        = d_volumes[0];
		TYPE *d_p         = d_volumes[1];
		TYPE *d_vx        = d_volumes[2];
		TYPE *d_vy        = d_volumes[3];
		TYPE *d_vz        = d_volumes[4];
		TYPE *d_difflux_E = d_volumes[5];

		dim3 block( TILE_DIMX, TILE_DIMY );
		dim3 grid( nx/block.x, ny/block.y );

//		cudaEventRecord( start, 0 );
		for( int i=0; i<num_iterations; i++ )
		{
			update_rE<TILE_DIMX,TILE_DIMY,RADIUS><<<grid,block>>>( 
				d_rE,
				d_p,
				d_vx, d_vy, d_vz,
				d_difflux_E,
				delta_t,
				nx, ny, nz,
				nx_pad, ny_pad, nz_pad );
		}
//		cudaEventRecord( stop, 0 );

		cudaDeviceSynchronize();
		error = cudaGetLastError();
		process_error( error, "update-rE kernel" );

/*		error = cudaEventElapsedTime( &elapsed_time_ms, start, stop );
		process_error( error, "get kernel time" );
	
		elapsed_time_ms /= num_iterations;
		throughput_mcells = 1e-3f * nx*ny*nz / elapsed_time_ms;

		printf( "%15s:  %7.2f ms, %7.1f MC/s\n", "update_rE", elapsed_time_ms, throughput_mcells );
*/
	}
#endif

#ifdef TRACE_UPDATE_U
	{	// update momenta, march in z, 8 volumes

		TYPE *d_ux        = d_volumes[0];
		TYPE *d_uy        = d_volumes[1];
		TYPE *d_uz        = d_volumes[2];
		TYPE *d_rho       = d_volumes[3];
		TYPE *d_difflux_x = d_volumes[4];
		TYPE *d_difflux_y = d_volumes[5];
		TYPE *d_difflux_z = d_volumes[6];
		TYPE *d_p         = d_volumes[7];

		dim3 block( TILE_DIMX, TILE_DIMY );
		dim3 grid( nx/block.x, ny/block.y );

//		cudaEventRecord( start, 0 );
		for( int i=0; i<num_iterations; i++ )
		{
			update_xyz_m4s<TILE_DIMX,TILE_DIMY,RADIUS><<<grid,block>>>( 
				d_ux, d_uy, d_uz,
				d_rho, d_p,
				d_difflux_x, d_difflux_y, d_difflux_z,
				delta_t,
				nx, ny, nz,
				nx_pad, ny_pad, nz_pad );
		}
//		cudaEventRecord( stop, 0 );

		cudaDeviceSynchronize();
		error = cudaGetLastError();
		process_error( error, "update-momentum m4s kernel" );

/*		error = cudaEventElapsedTime( &elapsed_time_ms, start, stop );
		process_error( error, "get kernel time" );
	
		elapsed_time_ms /= num_iterations;
		throughput_mcells = 1e-3f * nx*ny*nz / elapsed_time_ms;

		printf( "%15s:  %7.2f ms, %7.1f MC/s\n", "update_xyz_m4", elapsed_time_ms, throughput_mcells );
*/
	}
#endif

#ifdef TRACE_UPDATE_U_ALT
	{	// update momenta, march in z, 8 volumes

		TYPE *d_ux        = d_volumes[0];
		TYPE *d_uy        = d_volumes[1];
		TYPE *d_uz        = d_volumes[2];
		TYPE *d_rho       = d_volumes[3];
		TYPE *d_difflux_x = d_volumes[4];
		TYPE *d_difflux_y = d_volumes[5];
		TYPE *d_difflux_z = d_volumes[6];
		TYPE *d_p         = d_volumes[7];

		dim3 block( TILE_DIMX, TILE_DIMY );
		dim3 grid( nx/block.x, ny/block.y );

//		cudaEventRecord( start, 0 );
		for( int i=0; i<num_iterations; i++ )
		{
			update_xyz_m4s_p<TILE_DIMX,TILE_DIMY,RADIUS><<<grid,block>>>( 
				d_ux, d_uy, d_uz,
				d_rho, d_p,
				d_difflux_x, d_difflux_y, d_difflux_z,
				delta_t,
				nx, ny, nz,
				nx_pad, ny_pad, nz_pad );
		}
//		cudaEventRecord( stop, 0 );

		cudaDeviceSynchronize();
		error = cudaGetLastError();
		process_error( error, "update-momentum m4s kernel" );

/*		error = cudaEventElapsedTime( &elapsed_time_ms, start, stop );
		process_error( error, "get kernel time" );
	
		elapsed_time_ms /= num_iterations;
		throughput_mcells = 1e-3f * nx*ny*nz / elapsed_time_ms;

		printf( "%15s:  %7.2f ms, %7.1f MC/s\n", "update_xyz_m4", elapsed_time_ms, throughput_mcells );
*/
	}
#endif

	
	/////////////////////////////////////////
	// free GPU resources
	//

	for( int i=0; i<NUM_VOLS; i++ )
	{
		d_volumes[i] -= padding_to_first_output_cell;
		
		error = cudaFree( d_volumes[i] );
		char message[20];
		sprintf( message, "free vol %d", i );
		process_error( error, message );
	}

	error = cudaDeviceReset();
	process_error( error, "destroy GPU context" );

	return 0;
}

/*
#ifdef TRACE_UPDATE_RHO
	{	// update rho, march in z, 4 volumes

		TYPE *d_rho        = d_volumes[0];
		TYPE *d_momentum_x = d_volumes[1];
		TYPE *d_momentum_y = d_volumes[2];
		TYPE *d_momentum_z = d_volumes[3];

		dim3 block( TILE_DIMX, TILE_DIMY );
		dim3 grid( nx/block.x, ny/block.y );

		cudaEventRecord( start, 0 );
		for( int i=0; i<num_iterations; i++ )
		{
			update_rho_single_pass<TILE_DIMX,TILE_DIMY,RADIUS><<<grid,block>>>( 
				d_rho, d_momentum_x, d_momentum_y, d_momentum_z, 
				delta_t,
				nx, ny, nz,
				nx_pad, ny_pad, nz_pad );
		}
		cudaEventRecord( stop, 0 );

		cudaDeviceSynchronize();
		error = cudaGetLastError();
		process_error( error, "rho kernel" );

		error = cudaEventElapsedTime( &elapsed_time_ms, start, stop );
		process_error( error, "get kernel time" );
	
		elapsed_time_ms /= num_iterations;
		throughput_mcells = 1e-3f * nx*ny*nz / elapsed_time_ms;

		printf( "%17s:  %7.2f ms, %7.1f MC/s\n", "rho", elapsed_time_ms, throughput_mcells );
	}
#endif

#ifdef TRACE_DX
	{	// derivative in x, march in z, 2 volumes

		TYPE *d_vx_x = d_volumes[0];
		TYPE *d_vx   = d_volumes[1];

		dim3 block( TILE_DIMX, TILE_DIMY );
		dim3 grid( nx/block.x, ny/block.y );

		cudaEventRecord( start, 0 );
		for( int i=0; i<num_iterations; i++ )
		{
			first_derivative_x<TILE_DIMX,TILE_DIMY,RADIUS><<<grid,block>>>( 
				d_vx_x, d_vx,
				nx, ny, nz,
				nx_pad, ny_pad, nz_pad );
		}
		cudaEventRecord( stop, 0 );

		cudaDeviceSynchronize();
		error = cudaGetLastError();
		process_error( error, "x kernel" );

		error = cudaEventElapsedTime( &elapsed_time_ms, start, stop );
		process_error( error, "get kernel time" );
	
		elapsed_time_ms /= num_iterations;
		throughput_mcells = 1e-3f * nx*ny*nz / elapsed_time_ms;

		printf( "%17s:  %7.2f ms, %7.1f MC/s\n", "vx_x", elapsed_time_ms, throughput_mcells );
	}
#endif

#ifdef TRACE_DZ
	{	// derivative in z, march in z, 2 volumes

		TYPE *d_vz_z = d_volumes[0];
		TYPE *d_vz  = d_volumes[1];

		dim3 block( TILE_DIMX, TILE_DIMY );
		dim3 grid( nx/block.x, ny/block.y );

		cudaEventRecord( start, 0 );
		for( int i=0; i<num_iterations; i++ )
		{
			first_derivative_z<TILE_DIMX,TILE_DIMY,RADIUS><<<grid,block>>>( 
				d_vz_z, d_vz,
				nx, ny, nz,
				nx_pad, ny_pad, nz_pad );
		}
		cudaEventRecord( stop, 0 );

		cudaDeviceSynchronize();
		error = cudaGetLastError();
		process_error( error, "z kernel" );

		error = cudaEventElapsedTime( &elapsed_time_ms, start, stop );
		process_error( error, "get kernel time" );
	
		elapsed_time_ms /= num_iterations;
		throughput_mcells = 1e-3f * nx*ny*nz / elapsed_time_ms;

		printf( "%17s:  %7.2f ms, %7.1f MC/s\n", "vz_z", elapsed_time_ms, throughput_mcells );
	}
#endif

#ifdef TRACE_DIFFLUX_UX
	{	// difflux momentum-x, march in z, 4 volumes

		TYPE *d_difflux_x = d_volumes[0];
		TYPE *d_vx        = d_volumes[1];
		TYPE *d_vy_y      = d_volumes[2];
		TYPE *d_vz_z      = d_volumes[3];

		dim3 block( TILE_DIMX, TILE_DIMY );
		dim3 grid( nx/block.x, ny/block.y );

		cudaEventRecord( start, 0 );
		for( int i=0; i<num_iterations; i++ )
		{
			D_momentum_x<TILE_DIMX,TILE_DIMY,RADIUS><<<grid,block>>>( 
				d_difflux_x, d_vx, d_vy_y, d_vz_z,
				nx, ny, nz,
				nx_pad, ny_pad, nz_pad );
		}
		cudaEventRecord( stop, 0 );

		cudaDeviceSynchronize();
		error = cudaGetLastError();
		process_error( error, "difflux-x kernel" );

		error = cudaEventElapsedTime( &elapsed_time_ms, start, stop );
		process_error( error, "get kernel time" );
	
		elapsed_time_ms /= num_iterations;
		throughput_mcells = 1e-3f * nx*ny*nz / elapsed_time_ms;

		printf( "%17s:  %7.2f ms, %7.1f MC/s\n", "difflux_x", elapsed_time_ms, throughput_mcells );
	}
#endif

#ifdef TRACE_DIFFLUX_UY
	{	// difflux momentum-y, march in y, 4 volumes

		TYPE *d_difflux_y = d_volumes[0];
		TYPE *d_vy        = d_volumes[1];
		TYPE *d_vx_x      = d_volumes[2];
		TYPE *d_vz_z      = d_volumes[3];

		dim3 block( TILE_DIMX, TILE_DIMY );
		dim3 grid( nx/block.x, nz/block.y );

		cudaEventRecord( start, 0 );
		for( int i=0; i<num_iterations; i++ )
		{
			D_momentum_y_Q<TILE_DIMX,TILE_DIMY,RADIUS><<<grid,block>>>( 
				d_difflux_y, d_vy, d_vx_x, d_vz_z,
				nx, ny, nz,
				nx_pad, ny_pad, nz_pad );
		}
		cudaEventRecord( stop, 0 );

		cudaDeviceSynchronize();
		error = cudaGetLastError();
		process_error( error, "difflux-yQ kernel" );

		error = cudaEventElapsedTime( &elapsed_time_ms, start, stop );
		process_error( error, "get kernel time" );
	
		elapsed_time_ms /= num_iterations;
		throughput_mcells = 1e-3f * nx*ny*nz / elapsed_time_ms;

		printf( "%17s:  %7.2f ms, %7.1f MC/s\n", "difflux_yQ", elapsed_time_ms, throughput_mcells );
	}
#endif

#ifdef TRACE_DIFFLUX_UZ
	{	// difflux momentum-x, march in z, 4 volumes

		TYPE *d_difflux_z = d_volumes[0];
		TYPE *d_vz        = d_volumes[1];
		TYPE *d_vx_x      = d_volumes[2];
		TYPE *d_vy_y      = d_volumes[3];

		dim3 block( TILE_DIMX, TILE_DIMY );
		dim3 grid( nx/block.x, ny/block.y );

		cudaEventRecord( start, 0 );
		for( int i=0; i<num_iterations; i++ )
		{
			D_momentum_z_Q<TILE_DIMX,TILE_DIMY,RADIUS><<<grid,block>>>( 
				d_difflux_z, d_vz, d_vx_x, d_vy_y,
				nx, ny, nz,
				nx_pad, ny_pad, nz_pad );
		}
		cudaEventRecord( stop, 0 );

		cudaDeviceSynchronize();
		error = cudaGetLastError();
		process_error( error, "difflux-zQ kernel" );

		error = cudaEventElapsedTime( &elapsed_time_ms, start, stop );
		process_error( error, "get kernel time" );
	
		elapsed_time_ms /= num_iterations;
		throughput_mcells = 1e-3f * nx*ny*nz / elapsed_time_ms;

		printf( "%17s:  %7.2f ms, %7.1f MC/s\n", "difflux_zQ", elapsed_time_ms, throughput_mcells );
	}
#endif

#ifdef TRACE_DIFFLUX_E
	{	// difflux energy, march in z, 8 volumes

		TYPE *d_difflux_E = d_volumes[0];
		TYPE *d_vx        = d_volumes[1];
		TYPE *d_vy        = d_volumes[2];
		TYPE *d_vz        = d_volumes[3];
		TYPE *d_difflux_x = d_volumes[4];
		TYPE *d_difflux_y = d_volumes[5];
		TYPE *d_difflux_z = d_volumes[6];
		TYPE *d_T         = d_volumes[7];

		dim3 block( TILE_DIMX, TILE_DIMY );
		dim3 grid( nx/block.x, ny/block.y );

		TYPE alam = 5.3;
		TYPE eta  = 7.1;

		cudaEventRecord( start, 0 );
		for( int i=0; i<num_iterations; i++ )
		{
			D_energy_4q<TILE_DIMX,TILE_DIMY,RADIUS><<<grid,block>>>( 
				d_difflux_E, 
				alam, eta,
				d_T,
				d_vx, d_vy, d_vz,
				d_difflux_x, d_difflux_y, d_difflux_z,
				nx, ny, nz,
				nx_pad, ny_pad, nz_pad );
		}
		cudaEventRecord( stop, 0 );

		cudaDeviceSynchronize();
		error = cudaGetLastError();
		process_error( error, "difflux-E_4q kernel" );

		error = cudaEventElapsedTime( &elapsed_time_ms, start, stop );
		process_error( error, "get kernel time" );
	
		elapsed_time_ms /= num_iterations;
		throughput_mcells = 1e-3f * nx*ny*nz / elapsed_time_ms;

		printf( "%17s:  %7.2f ms, %7.1f MC/s\n", "difflux_E_4q", elapsed_time_ms, throughput_mcells );
	}
#endif

#ifdef TRACE_UPDATE_E
	{	// update energy, march in z, 6 volumes

		TYPE *d_rE        = d_volumes[0];
		TYPE *d_p         = d_volumes[1];
		TYPE *d_vx        = d_volumes[2];
		TYPE *d_vy        = d_volumes[3];
		TYPE *d_vz        = d_volumes[4];
		TYPE *d_difflux_E = d_volumes[5];

		dim3 block( TILE_DIMX, TILE_DIMY );
		dim3 grid( nx/block.x, ny/block.y );

		cudaEventRecord( start, 0 );
		for( int i=0; i<num_iterations; i++ )
		{
			update_rE<TILE_DIMX,TILE_DIMY,RADIUS><<<grid,block>>>( 
				d_rE,
				d_p,
				d_vx, d_vy, d_vz,
				d_difflux_E,
				delta_t,
				nx, ny, nz,
				nx_pad, ny_pad, nz_pad );
		}
		cudaEventRecord( stop, 0 );

		cudaDeviceSynchronize();
		error = cudaGetLastError();
		process_error( error, "update-rE kernel" );

		error = cudaEventElapsedTime( &elapsed_time_ms, start, stop );
		process_error( error, "get kernel time" );
	
		elapsed_time_ms /= num_iterations;
		throughput_mcells = 1e-3f * nx*ny*nz / elapsed_time_ms;

		printf( "%17s:  %7.2f ms, %7.1f MC/s\n", "update_rE", elapsed_time_ms, throughput_mcells );
	}
#endif

#ifdef TRACE_UPDATE_U
	{	// update momenta, march in z, 8 volumes

		TYPE *d_ux        = d_volumes[0];
		TYPE *d_uy        = d_volumes[1];
		TYPE *d_uz        = d_volumes[2];
		TYPE *d_rho       = d_volumes[3];
		TYPE *d_difflux_x = d_volumes[4];
		TYPE *d_difflux_y = d_volumes[5];
		TYPE *d_difflux_z = d_volumes[6];
		TYPE *d_p         = d_volumes[7];

		dim3 block( TILE_DIMX, TILE_DIMY );
		dim3 grid( nx/block.x, ny/block.y );

		cudaEventRecord( start, 0 );
		for( int i=0; i<num_iterations; i++ )
		{
			update_xyz_m4s<TILE_DIMX,TILE_DIMY,RADIUS><<<grid,block>>>( 
				d_ux, d_uy, d_uz,
				d_rho, d_p,
				d_difflux_x, d_difflux_y, d_difflux_z,
				delta_t,
				nx, ny, nz,
				nx_pad, ny_pad, nz_pad );
		}
		cudaEventRecord( stop, 0 );

		cudaDeviceSynchronize();
		error = cudaGetLastError();
		process_error( error, "update-momentum m4s kernel" );

		error = cudaEventElapsedTime( &elapsed_time_ms, start, stop );
		process_error( error, "get kernel time" );
	
		elapsed_time_ms /= num_iterations;
		throughput_mcells = 1e-3f * nx*ny*nz / elapsed_time_ms;

		printf( "%17s:  %7.2f ms, %7.1f MC/s\n", "update_xyz_m4", elapsed_time_ms, throughput_mcells );
	}
#endif

#ifdef TRACE_UPDATE_U
	{	// update momenta, march in z, 8 volumes

		TYPE *d_ux        = d_volumes[0];
		TYPE *d_uy        = d_volumes[1];
		TYPE *d_uz        = d_volumes[2];
		TYPE *d_rho       = d_volumes[3];
		TYPE *d_difflux_x = d_volumes[4];
		TYPE *d_difflux_y = d_volumes[5];
		TYPE *d_difflux_z = d_volumes[6];
		TYPE *d_p         = d_volumes[7];

		dim3 block( TILE_DIMX, TILE_DIMY );
		dim3 grid( nx/block.x, ny/block.y );

		cudaEventRecord( start, 0 );
		for( int i=0; i<num_iterations; i++ )
		{
			update_xyz_m4s_p<TILE_DIMX,TILE_DIMY,RADIUS><<<grid,block>>>( 
				d_ux, d_uy, d_uz,
				d_rho, d_p,
				d_difflux_x, d_difflux_y, d_difflux_z,
				delta_t,
				nx, ny, nz,
				nx_pad, ny_pad, nz_pad );
		}
		cudaEventRecord( stop, 0 );

		cudaDeviceSynchronize();
		error = cudaGetLastError();
		process_error( error, "update-momentum m4s kernel" );

		error = cudaEventElapsedTime( &elapsed_time_ms, start, stop );
		process_error( error, "get kernel time" );
	
		elapsed_time_ms /= num_iterations;
		throughput_mcells = 1e-3f * nx*ny*nz / elapsed_time_ms;

		printf( "%17s:  %7.2f ms, %7.1f MC/s\n", "update_xyz_m4_p", elapsed_time_ms, throughput_mcells );
	}
#endif
	
	
	/////////////////////////////////////////
	// free GPU resources
	//

	for( int i=0; i<NUM_VOLS; i++ )
	{
		d_volumes[i] -= padding_to_first_output_cell;
		
		error = cudaFree( d_volumes[i] );
		char message[20];
		sprintf( message, "free vol %d", i );
		process_error( error, message );
	}

	error = cudaDeviceReset();
	process_error( error, "destroy GPU context" );

	return 0;
}
*/