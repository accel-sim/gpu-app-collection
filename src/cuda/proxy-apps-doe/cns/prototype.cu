#include <stdio.h>


//#define TRACE_DX
//#define TRACE_DZ
//#define TRACE_DIFFLUX_UX
//#define TRACE_DIFFLUX_UY
//#define TRACE_DIFFLUX_UZ
//#define TRACE_DIFFLUX_E
//#define TRACE_UPDATE_RHO
//#define TRACE_UPDATE_E
//#define TRACE_UPDATE_U

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
	__shared__ TYPE s_ux[tile_dimy][tile_dimx+2*radius];
	__shared__ TYPE s_uy[tile_dimy+2*radius][tile_dimx];

	int ix = blockIdx.x * tile_dimx + threadIdx.x;
	int iy = blockIdx.y * tile_dimy + threadIdx.y;

	const int tx = threadIdx.x + radius;
	const int ty = threadIdx.y + radius;

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

		__syncthreads();
		
		s_ux[threadIdx.y][tx] = __ldg( &g_ux[idx_out] );
		s_uy[ty][threadIdx.x] = __ldg( &g_uy[idx_out] );

		if( threadIdx.x < radius )
		{
			s_ux[threadIdx.y][tx-radius]    = __ldg( &g_ux[idx_out-radius] );
			s_ux[threadIdx.y][tx+tile_dimx] = __ldg( &g_ux[idx_out+tile_dimx] );
		}
		
		if( threadIdx.y < radius )
		{
			s_uy[ty-radius][threadIdx.x]    = __ldg( &g_uy[idx_out - radius*nx_pad] );
			s_uy[ty+tile_dimy][threadIdx.x] = __ldg( &g_uy[idx_out + tile_dimy*nx_pad] );
		}
		__syncthreads();

		TYPE del_momenta = 0;
				
		#pragma unroll
		for( int i=1; i<=radius; i++ )
		{
			del_momenta += c_first[i] * (   s_ux[threadIdx.y][tx+i] - s_ux[threadIdx.y][tx-i]
			                              + s_uy[ty+i][threadIdx.x] - s_uy[ty-i][threadIdx.x]
			                              + queue[radius+i]         - queue[radius-i]         );
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
	__shared__ TYPE s_field[tile_dimy][tile_dimx+2*radius];
	
	int ix = blockIdx.x * tile_dimx + threadIdx.x;
	int iy = blockIdx.y * tile_dimy + threadIdx.y;
	int stride = nx_pad * ny_pad;

	int idx = iy * nx_pad + ix;
	int tx = threadIdx.x + radius;

	OUTER_UNROLL
	for( int iz=0; iz<nz; iz++ )
	{
		__syncthreads();
		s_field[threadIdx.y][tx] = __ldg( &g_field[idx] );
		if( threadIdx.x < radius )
		{
			s_field[threadIdx.y][tx-radius]    = __ldg( &g_field[idx-radius]    );
			s_field[threadIdx.y][tx+tile_dimx] = __ldg( &g_field[idx+tile_dimx] );
		}
		__syncthreads();

		TYPE deriv = c_first[0] * s_field[threadIdx.y][tx];
		#pragma unroll
		for( int i=1; i<=radius; i++ )
			deriv += c_first[i] * ( s_field[threadIdx.y][tx+i] - s_field[threadIdx.y][tx-i] );

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
	__shared__ TYPE s_vx[tile_dimy+2*radius][tile_dimx+2*radius];
	__shared__ TYPE s_vy_y[tile_dimy][tile_dimx+2*radius];
	__shared__ TYPE s_vz_z[tile_dimy][tile_dimx+2*radius];
	
	int ix = blockIdx.x * tile_dimx + threadIdx.x;
	int iy = blockIdx.y * tile_dimy + threadIdx.y;
	int stride = nx_pad * ny_pad;

	int idx_out = iy * nx_pad + ix;
	int idx_in  = idx_out - radius * stride;

	int tx = threadIdx.x + radius;
	int ty = threadIdx.y + radius;
	
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

		__syncthreads();

		s_vx[ty][tx]            = queue_vx[radius];
		s_vy_y[threadIdx.y][tx] = __ldg( &g_vy_y[idx_out] );
		s_vz_z[threadIdx.y][tx] = __ldg( &g_vz_z[idx_out] );

		if( threadIdx.x < radius )
		{
			s_vx[ty][tx-radius]               = __ldg( &g_vx[idx_out-radius]    );
			s_vx[ty][tx+tile_dimx]            = __ldg( &g_vx[idx_out+tile_dimx] );

			s_vy_y[threadIdx.y][tx-radius]    = __ldg( &g_vy_y[idx_out-radius]    );
			s_vy_y[threadIdx.y][tx+tile_dimx] = __ldg( &g_vy_y[idx_out+tile_dimx] );

			s_vz_z[threadIdx.y][tx-radius]    = __ldg( &g_vz_z[idx_out-radius]    );
			s_vz_z[threadIdx.y][tx+tile_dimx] = __ldg( &g_vz_z[idx_out+tile_dimx] );
		}

		if( threadIdx.y < radius )
		{
			s_vx[ty-radius][tx]               = __ldg( &g_vx[idx_out-radius*nx_pad] );
			s_vx[ty+tile_dimy][tx]            = __ldg( &g_vx[idx_out+tile_dimy*nx_pad] );
		}
		
		__syncthreads();

		TYPE vx_xx, vx_yy, vx_zz, vy_xy, vz_xz;

		vx_xx = vx_yy = vx_zz = c_second[0] * queue_vx[radius];
		vy_xy = c_first[0] * s_vy_y[threadIdx.y][tx];
		vz_xz = c_first[0] * s_vz_z[threadIdx.y][tx];

		#pragma unroll
		for( int i=1; i<=radius; i++ )
		{
			vx_xx += c_second[i] * ( s_vx[ty][tx+radius] + s_vx[ty][tx-radius] );
			vx_yy += c_second[i] * ( s_vx[ty+radius][tx] + s_vx[ty-radius][tx] );
			vx_zz += c_second[i] * ( queue_vx[radius+i]  + queue_vx[radius-i]  );

			vy_xy += c_first[i] * ( s_vy_y[threadIdx.y][tx+radius] - s_vy_y[threadIdx.y][tx-radius] );
			vz_xz += c_first[i] * ( s_vz_z[threadIdx.y][tx+radius] - s_vz_z[threadIdx.y][tx-radius] );
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

	__shared__ TYPE s_vy[tile_dimy+2*radius][tile_dimx+2*radius];
		
	int ix = blockIdx.x * tile_dimx + threadIdx.x;
	int iz = blockIdx.y * tile_dimy + threadIdx.y;
	int stride = nx_pad;

	int idx_out = iz * (nx_pad*ny_pad) + ix;
	int idx_in  = idx_out - radius * stride;

	int tx = threadIdx.x + radius;
	int ty = threadIdx.y + radius;
	
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

		__syncthreads();

		s_vy[ty][tx] = queue_vy[radius];
		
		if( threadIdx.x < radius )
		{
			s_vy[ty][tx-radius]               = __ldg( &g_vy[idx_out-radius]    );
			s_vy[ty][tx+tile_dimx]            = __ldg( &g_vy[idx_out+tile_dimx] );
		}

		if( threadIdx.y < radius )
		{
			s_vy[ty-radius][tx]               = __ldg( &g_vy[idx_out - radius   *nx_pad*ny_pad] );
			s_vy[ty+tile_dimy][tx]            = __ldg( &g_vy[idx_out + tile_dimy*nx_pad*ny_pad] );
		}
		
		__syncthreads();

		TYPE vy_xx, vy_yy, vy_zz, vx_xy, vz_yz;
		
		vy_xx = vy_yy = vy_zz = c_second[0] * queue_vy[radius];
		vx_xy = c_first[0] * queue_vx_x[radius];
		vz_yz = c_first[0] * queue_vz_z[radius];

		#pragma unroll
		for( int i=1; i<=radius; i++ )
		{
			vy_xx += c_second[i] * ( s_vy[ty][tx+i] + s_vy[ty][tx-i] );
			vy_yy += c_second[i] * ( s_vy[ty+i][tx] + s_vy[ty-i][tx] );
			vy_zz += c_second[i] * ( queue_vy[radius+i] + queue_vy[radius-i] );

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

	__shared__ TYPE s_vz[tile_dimy+2*radius][tile_dimx+2*radius];

	int ix = blockIdx.x * tile_dimx + threadIdx.x;
	int iy = blockIdx.y * tile_dimy + threadIdx.y;
	int stride = nx_pad * ny_pad;

	int idx_out = iy * nx_pad + ix;
	int idx_in  = idx_out - radius * stride;

	int tx = threadIdx.x + radius;
	int ty = threadIdx.y + radius;
	
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

		__syncthreads();

		s_vz[ty][tx] = queue_vz[radius];
		
		if( threadIdx.x < radius )
		{
			s_vz[ty][tx-radius]    = __ldg( &g_vz[idx_out-radius]    );
			s_vz[ty][tx+tile_dimx] = __ldg( &g_vz[idx_out+tile_dimx] );
		}

		if( threadIdx.y < radius )
		{
			s_vz[ty-radius][tx]    = __ldg( &g_vz[idx_out - radius   *nx_pad] );
			s_vz[ty+tile_dimy][tx] = __ldg( &g_vz[idx_out + tile_dimy*nx_pad] );
		}
		
		__syncthreads();

		TYPE vz_xx, vz_yy, vz_zz, vx_xz, vy_yz;
		
		vz_xx = vz_yy = vz_zz = c_second[0] * queue_vz[radius];
		vx_xz = c_first[0] * queue_vx_x[radius];
		vy_yz = c_first[0] * queue_vy_y[radius];

		#pragma unroll
		for( int i=1; i<=radius; i++ )
		{
			vz_xx += c_second[i] * ( s_vz[ty][tx+i] + s_vz[ty][tx-i] );
			vz_yy += c_second[i] * ( queue_vz[radius+i] + queue_vz[radius-i] );
			vz_zz += c_second[i] * ( s_vz[ty+i][tx] + s_vz[ty-i][tx] );

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

	__shared__ TYPE s_T[tile_dimy+2*radius][tile_dimx+2*radius];
	__shared__ TYPE s_vx[tile_dimy+2*radius][tile_dimx+2*radius];
	__shared__ TYPE s_vy[tile_dimy+2*radius][tile_dimx+2*radius];
	__shared__ TYPE s_vz[tile_dimy+2*radius][tile_dimx+2*radius];
		
	int ix = blockIdx.x * tile_dimx + threadIdx.x;
	int iy = blockIdx.y * tile_dimy + threadIdx.y;
	int stride = nx_pad*ny_pad;

	int idx_out = iy * nx_pad + ix;
	int idx_in  = idx_out - radius * stride;

	int tx = threadIdx.x + radius;
	int ty = threadIdx.y + radius;
	
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

		__syncthreads();

		s_T[ty][tx]  = queue_T[radius];
		s_vx[ty][tx] = queue_vx[radius];
		s_vy[ty][tx] = queue_vy[radius];
		s_vz[ty][tx] = queue_vz[radius];
		
		if( threadIdx.x < radius )
		{
			s_T[ty][tx-radius]    = __ldg( &g_T[idx_out - radius]    );
			s_T[ty][tx+tile_dimx] = __ldg( &g_T[idx_out + tile_dimx] );

			s_vx[ty][tx-radius]    = __ldg( &g_vx[idx_out - radius]    );
			s_vx[ty][tx+tile_dimx] = __ldg( &g_vx[idx_out + tile_dimx] );

			s_vy[ty][tx-radius]    = __ldg( &g_vy[idx_out - radius]    );
			s_vy[ty][tx+tile_dimx] = __ldg( &g_vy[idx_out + tile_dimx] );

			s_vz[ty][tx-radius]    = __ldg( &g_vz[idx_out - radius]    );
			s_vz[ty][tx+tile_dimx] = __ldg( &g_vz[idx_out + tile_dimx] );
		}

		if( threadIdx.y < radius )
		{
			s_T[ty-radius][tx]    = __ldg( &g_T[idx_out - radius*nx_pad]    );
			s_T[ty+tile_dimy][tx] = __ldg( &g_T[idx_out + tile_dimy*nx_pad] );

			s_vx[ty-radius][tx]    = __ldg( &g_vx[idx_out - radius*nx_pad]    );
			s_vx[ty+tile_dimy][tx] = __ldg( &g_vx[idx_out + tile_dimy*nx_pad] );

			s_vy[ty-radius][tx]    = __ldg( &g_vy[idx_out - radius*nx_pad]    );
			s_vy[ty+tile_dimy][tx] = __ldg( &g_vy[idx_out + tile_dimy*nx_pad] );

			s_vz[ty-radius][tx]    = __ldg( &g_vz[idx_out - radius*nx_pad]    );
			s_vz[ty+tile_dimy][tx] = __ldg( &g_vz[idx_out + tile_dimy*nx_pad] );
		}
		
		__syncthreads();

		TYPE vx_x=0, vx_y=0, vx_z=0;
		TYPE vy_x=0, vy_y=0, vy_z=0;
		TYPE vz_x=0, vz_y=0, vz_z=0;
		
		TYPE laplacian_T = 3 * c_second[0] * queue_T[radius];

		#pragma unroll
		for( int i=1; i<=radius; i++ )
		{
			laplacian_T += c_second[i] * ( queue_T[radius+i] + queue_T[radius-i] +
				                           s_T[ty][tx+i] + s_T[ty][tx-i] +
										   s_T[ty+i][tx] + s_T[ty-i][tx] );

			vx_x += c_first[i] * ( s_vx[ty][tx+i] + s_vx[ty][tx-i] );
			vx_y += c_first[i] * ( s_vx[ty+i][tx] + s_vx[ty-i][tx] );
			vx_z += c_first[i] * ( queue_vx[radius+i] + queue_vx[radius-i] );
			
			vy_x += c_first[i] * ( s_vy[ty][tx+i] + s_vy[ty][tx-i] );
			vy_y += c_first[i] * ( s_vy[ty+i][tx] + s_vy[ty-i][tx] );
			vy_z += c_first[i] * ( queue_vy[radius+i] + queue_vy[radius-i] );

			vz_x += c_first[i] * ( s_vz[ty][tx+i] + s_vz[ty][tx-i] );
			vz_y += c_first[i] * ( s_vz[ty+i][tx] + s_vz[ty-i][tx] );
			vz_z += c_first[i] * ( queue_vz[radius+i] + queue_vz[radius-i] );
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
	__shared__ TYPE s_rEp_vx[tile_dimy][tile_dimx+2*radius];
	__shared__ TYPE s_rEp_vy[tile_dimy+2*radius][tile_dimx];

	int ix = blockIdx.x * tile_dimx + threadIdx.x;
	int iy = blockIdx.y * tile_dimy + threadIdx.y;

	const int tx = threadIdx.x + radius;
	const int ty = threadIdx.y + radius;

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

		__syncthreads();
		
		s_rEp_vx[threadIdx.y][tx] = __ldg( &g_vx[idx_out] ) * queue_rE_p[radius];
		s_rEp_vy[ty][threadIdx.x] = __ldg( &g_vy[idx_out] ) * queue_rE_p[radius];

		if( threadIdx.x < radius )
		{
			s_rEp_vx[threadIdx.y][tx-radius]    = __ldg( &g_vx[idx_out - radius] )    * ( __ldg( &g_rE[idx_out - radius] )    + __ldg( &g_p[idx_out - radius] )   );
			s_rEp_vx[threadIdx.y][tx+tile_dimx] = __ldg( &g_vx[idx_out + tile_dimx] ) * ( __ldg( &g_rE[idx_out + tile_dimx] ) + __ldg( &g_p[idx_out + tile_dimx]) );
		}

		if( threadIdx.y < radius )
		{
			s_rEp_vy[ty-radius][threadIdx.x]    = __ldg( &g_vy[idx_out - radius*nx_pad] )    * ( __ldg( &g_rE[idx_out - radius*nx_pad]  )   + __ldg( &g_p[idx_out - radius*nx_pad] )   );
			s_rEp_vy[ty+tile_dimy][threadIdx.x] = __ldg( &g_vy[idx_out + tile_dimy*nx_pad] ) * ( __ldg( &g_rE[idx_out + tile_dimy*nx_pad] ) + __ldg( &g_p[idx_out + tile_dimy*nx_pad] ));
		}
		__syncthreads();

		TYPE rEp_vx_x = 0;
		TYPE rEp_vy_y = 0;
		TYPE rEp_vz_z = 0;
				
		#pragma unroll
		for( int i=1; i<=radius; i++ )
		{
			rEp_vx_x += c_first[i] * ( s_rEp_vx[threadIdx.y][tx+i] - s_rEp_vx[threadIdx.y][tx-i] );
			rEp_vy_y += c_first[i] * ( s_rEp_vy[ty+i][threadIdx.x] - s_rEp_vy[ty-i][threadIdx.x] );
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
	__shared__ TYPE s_ux[tile_dimy+2*radius][tile_dimx+2*radius];
	__shared__ TYPE s_uy[tile_dimy+2*radius][tile_dimx+2*radius];
	__shared__ TYPE s_uz[tile_dimy+2*radius][tile_dimx+2*radius];
	TYPE (*s_p)[tile_dimx+2*radius] = s_ux;
	__shared__ TYPE s_rho[tile_dimy+2*radius][tile_dimx+2*radius];
	
	TYPE l_ux[2*radius+1], l_uy[2*radius+1], l_uz[2*radius+1], l_rho[2*radius+1], l_p[2*radius+1];

	int ix = blockIdx.x * tile_dimx + threadIdx.x;
	int iy = blockIdx.y * tile_dimy + threadIdx.y;

	const int tx = threadIdx.x + radius;
	const int ty = threadIdx.y + radius;

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
		TYPE hypterm_x=0, hypterm_y=0, hypterm_z=0;
		
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
		
		if( threadIdx.x < radius )
		{
			s_p[ty][tx+tile_dimx]   = __ldg( &g_p[idx_out-radius+tile_dimx] );
			s_p[ty][tx+tile_dimx]   = __ldg( &g_p[idx_out-radius+tile_dimx] );
		}
		if( threadIdx.y < radius )
		{
			s_p[ty-radius][tx]      = __ldg( &g_p[idx_out - radius*nx_pad] );
			s_p[ty+tile_dimy][tx]   = __ldg( &g_p[idx_out + tile_dimy*nx_pad] );
		}
		__syncthreads();
		for( int i=1; i<=radius; i++ )
		{
			hypterm_x += c_first[i] * (	s_p[ty][tx+i] - s_p[ty][tx-i] );  // x-deriv of pressure
			hypterm_y += c_first[i] * ( s_p[ty+i][tx] - s_p[ty-i][tx] );  // y-deriv of pressure
		}
		__syncthreads();

		if( threadIdx.x < radius )
		{
			s_ux[ty][tx-radius]     = __ldg( &g_ux[idx_out-radius] );
			s_ux[ty][tx+tile_dimx]  = __ldg( &g_ux[idx_out-radius+tile_dimx] );
			s_uy[ty][tx-radius]     = __ldg( &g_uy[idx_out-radius] );
			s_uy[ty][tx+tile_dimx]  = __ldg( &g_uy[idx_out-radius+tile_dimx] );
			s_uz[ty][tx-radius]     = __ldg( &g_uz[idx_out-radius] );
			s_uz[ty][tx-radius]     = __ldg( &g_uz[idx_out-radius] );
			s_rho[ty][tx+tile_dimx] = __ldg( &g_rho[idx_out-radius+tile_dimx] );
			s_rho[ty][tx-radius]    = __ldg( &g_rho[idx_out-radius] );
		}
		if( threadIdx.y < radius )
		{
			s_ux[ty-radius][tx]     = __ldg( &g_ux[idx_out - radius*nx_pad] );
			s_ux[ty+tile_dimy][tx]  = __ldg( &g_ux[idx_out + tile_dimy*nx_pad] );
			s_uy[ty-radius][tx]     = __ldg( &g_uy[idx_out - radius*nx_pad] );
			s_uy[ty+tile_dimy][tx]  = __ldg( &g_uy[idx_out + tile_dimy*nx_pad] );
			s_uz[ty-radius][tx]     = __ldg( &g_uz[idx_out - radius*nx_pad] );
			s_uz[ty+tile_dimy][tx]  = __ldg( &g_uz[idx_out + tile_dimy*nx_pad] );
			s_rho[ty-radius][tx]    = __ldg( &g_rho[idx_out - radius*nx_pad] );
			s_rho[ty+tile_dimy][tx] = __ldg( &g_rho[idx_out + tile_dimy*nx_pad] );
		}
		__syncthreads();

				
		////////////////////////////////////
		// compute the x component
		//
		{		
			#pragma unroll
			for( int i=1; i<=radius; i++ )
			{
				hypterm_x += c_first[i] * (	
					( s_rho[ty][tx+i]*s_ux[ty][tx+i]*s_ux[ty][tx+i] - s_rho[ty][tx-i]*s_ux[ty][tx-i]*s_ux[ty][tx-i] ) +  // x-deriv of rho-ux-ux product
					( s_rho[ty+i][tx]*s_ux[ty+i][tx]*s_uy[ty+i][tx] - s_rho[ty-i][tx]*s_ux[ty-i][tx]*s_uy[ty-i][tx] ) +  // y-deriv of rho-ux-uy product
					( l_rho[radius+i]*l_ux[radius+i]*l_uz[radius+i] - l_rho[radius-i]*l_ux[radius-i]*l_uz[radius-i] )    // z-deriv of rho-ux-uz product
				);
			}

			g_ux[idx_out] = l_ux[radius] + __ldg( &g_difflux_x[idx_out] ) - delta_t * hypterm_x;
		}

		//__threadfence_block();

		////////////////////////////////////
		// compute the y component
		//
		{
			TYPE hypterm_y = 0;
				
			#pragma unroll
			for( int i=1; i<=radius; i++ )
			{
				hypterm_y += c_first[i] * (	
					( s_rho[ty][tx+i]*s_ux[ty][tx+i]*s_uy[ty][tx+i] - s_rho[ty][tx-i]*s_ux[ty][tx-i]*s_uy[ty][tx-i] ) +  // x-deriv of rho-ux-uy product
					( s_rho[ty+i][tx]*s_uy[ty+i][tx]*s_uy[ty+i][tx] - s_rho[ty-i][tx]*s_uy[ty-i][tx]*s_uy[ty-i][tx] ) +  // y-deriv of rho-uy-uy product
					( l_rho[radius+i]*l_uy[radius+i]*l_uz[radius+i] - l_rho[radius-i]*l_uy[radius-i]*l_uz[radius-i] )    // z-deriv of rho-uy-uz product
				);
			}

			g_uy[idx_out] = l_uy[radius] + __ldg( &g_difflux_y[idx_out] ) - delta_t * hypterm_y;
		}

		//__threadfence_block();

		////////////////////////////////////
		// compute the z component
		//
		{
			TYPE hypterm_z = 0;
				
			#pragma unroll
			for( int i=1; i<=radius; i++ )
			{
				hypterm_z += c_first[i] * (	
					( s_rho[ty][tx+i]*s_ux[ty][tx+i]*s_uz[ty][tx+i] - s_rho[ty][tx-i]*s_ux[ty][tx-i]*s_uz[ty][tx-i] ) +  // x-deriv of rho-ux-uz product
					( s_rho[ty+i][tx]*s_uy[ty+i][tx]*s_uz[ty+i][tx] - s_rho[ty-i][tx]*s_uy[ty-i][tx]*s_uz[ty-i][tx] ) +  // y-deriv of rho-uy-uz product
					( l_rho[radius+i]*l_uz[radius+i]*l_uz[radius+i] - l_rho[radius-i]*l_uz[radius-i]*l_uz[radius-i] ) +  // z-deriv of rho-uz-uz product
					( l_p[radius+i]                                 - l_p[radius-i] )                                    // z-deriv of pressure
				);
			}

			g_uz[idx_out] = l_uz[radius] + __ldg( &g_difflux_z[idx_out] ) - delta_t * hypterm_z;
		}
		
		__syncthreads();

		idx_in  += slice_stride;
		idx_out += slice_stride;
	}
}
#endif


#define NUM_VOLS 8

int main( int argc, char *argv[] )
{
	int nx = 2*32;
	int ny = 1*32;
	int nz = 1*32;
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
