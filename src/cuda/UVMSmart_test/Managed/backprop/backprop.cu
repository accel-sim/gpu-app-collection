
#include "backprop.h"
#include "backprop_cuda_kernel.cu"
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#define OPEN

#define ABS(x)          (((x) > 0.0) ? (x) : (-(x)))

#define fastcopy(to,from,len)\
{\
  register char *_to,*_from;\
  register int _i,_l;\
  _to = (char *)(to);\
  _from = (char *)(from);\
  _l = (len);\
  for (_i = 0; _i < _l; _i++) *_to++ = *_from++;\
}

#define BIGRND 0x7fffffff

#define GPU
#define THREADS 256
#define WIDTH 16  // shared memory width  
#define HEIGHT 16 // shared memory height

#define ETA 0.3       //eta value
#define MOMENTUM 0.3  //momentum value
#define NUM_THREAD 4  //OpenMP threads

#include <stdio.h>
#include <stdlib.h>

int layer_size = 0;
unsigned int num_threads = 0;
unsigned int num_blocks = 0;





/*** Allocate 1d array of floats ***/

float *alloc_1d_dbl(int n)
{
  float *new_arr;

  cudaMallocManaged(&new_arr, n * sizeof (float));
  if (new_arr == NULL) {
    printf("ALLOC_1D_DBL: Couldn't allocate array of %d floats\n", n);
    return (NULL);
  }
  return (new_arr);
}

void bpnn_copy(float *wf,float *wt,int m,int n)
{
  int i, j;

  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
      wt[i*(n+1)+j] =  wf[i*(n+1)+j];
    }
  }
}

void bpnn_randomize_weights(float *w,int m,int n)
{
  int i, j;

  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
      w[i*(n+1)+j] =  (float) rand()/RAND_MAX;
    }
  }
}

void bpnn_randomize_row(float *w,int m)
{
  int i;
  for (i = 0; i <= m; i++) {
    //w[i] = (float) rand()/RAND_MAX;
    w[i] = 0.1;
  }
}


void bpnn_zero_weights(float *w,int m,int n)
{
  int i, j;

  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
      w[i*(n+1)+j] =  0.0;
    }
  }
}


BPNN *bpnn_internal_create(int n_in,int n_hidden,int n_out)
{
  BPNN *newnet;

  newnet = (BPNN *) malloc (sizeof (BPNN));
  if (newnet == NULL) {
    printf("BPNN_CREATE: Couldn't allocate neural network\n");
    return (NULL);
  }

  newnet->input_n = n_in;
  newnet->hidden_n = n_hidden;
  newnet->output_n = n_out;
  newnet->input_units = alloc_1d_dbl(n_in + 1);
  newnet->hidden_units = alloc_1d_dbl(n_hidden + 1);
  newnet->output_units = alloc_1d_dbl(n_out + 1);

  newnet->hidden_delta = alloc_1d_dbl(n_hidden + 1);
  newnet->output_delta = alloc_1d_dbl(n_out + 1);
  newnet->target = alloc_1d_dbl(n_out + 1);

  newnet->input_weights = alloc_1d_dbl( (n_in + 1) * (n_hidden + 1) );
  newnet->input_weights2 = alloc_1d_dbl( (n_in + 1) * (n_hidden + 1) );
  newnet->hidden_weights = alloc_1d_dbl( (n_hidden + 1) * (n_out + 1) );

  newnet->input_prev_weights = alloc_1d_dbl( (n_in + 1) * (n_hidden + 1) );
  newnet->hidden_prev_weights = alloc_1d_dbl( (n_hidden + 1) * (n_out + 1) );

  return (newnet);
}


void bpnn_free(BPNN *net)
{
  cudaFree(net->input_units);
  cudaFree(net->hidden_units);
  cudaFree(net->output_units);

  cudaFree(net->hidden_delta);
  cudaFree(net->output_delta);
  cudaFree(net->target);

  cudaFree(net->input_weights);
  cudaFree(net->input_weights2);
  cudaFree(net->input_prev_weights);

  cudaFree(net->hidden_weights);
  cudaFree(net->hidden_prev_weights);

  free((char *) net);
}
                                                 


/*** Creates a new fully-connected network from scratch,
     with the given numbers of input, hidden, and output units.
     Threshold units are automatically included.  All weights are
     randomly initialized.

     Space is also allocated for temporary storage (momentum weights,
     error computations, etc).
***/

BPNN *bpnn_create(int n_in,int n_hidden,int n_out)
{

  BPNN *newnet;

  newnet = bpnn_internal_create(n_in, n_hidden, n_out);

#ifdef INITZERO
  bpnn_zero_weights(newnet->input_weights, n_in, n_hidden);
#else
  bpnn_randomize_weights(newnet->input_weights, n_in, n_hidden);
#endif
  bpnn_copy(newnet->input_weights, newnet->input_weights2,n_in,n_hidden);
  bpnn_randomize_weights(newnet->hidden_weights, n_hidden, n_out);
  bpnn_zero_weights(newnet->input_prev_weights, n_in, n_hidden);
  bpnn_zero_weights(newnet->hidden_prev_weights, n_hidden, n_out);
  bpnn_randomize_row(newnet->target, n_out);
  return (newnet);
}


void bpnn_layerforward(float* l1,float* l2,float* conn,int n1,int n2)
{
  float sum;
  int j, k;

  /*** Set up thresholding unit ***/
  l1[0] = 1.0;
#ifdef OPEN
  omp_set_num_threads(NUM_THREAD);
  #pragma omp parallel for shared(conn, n1, n2, l1) private(k, j) reduction(+: sum) schedule(static)
#endif
  /*** For each unit in second layer ***/
  for (j = 1; j <= n2; j++) {

    /*** Compute weighted sum of its inputs ***/
    sum = 0.0;
    for (k = 0; k <= n1; k++) {
      sum += conn[k*(n2+1)+j] * l1[k];
    }
    l2[j] = squash(sum);
  }
}

void bpnn_output_error(float* delta,float* target,float* output,int nj,float* err)
{
  int j;
  float o, t, errsum;
  errsum = 0.0;
  for (j = 1; j <= nj; j++) {
    o = output[j];
    t = target[j];
    delta[j] = o * (1.0 - o) * (t - o);
    errsum += ABS(delta[j]);
  }
  *err = errsum;
}


void bpnn_hidden_error(float* delta_h,
                        int                nh,
                        float*             delta_o,
                        int                no,
                        float*             who,
                        float*             hidden,
                        float*             err)
{
  int j, k;
  float h, sum, errsum;

  errsum = 0.0;
  for (j = 1; j <= nh; j++) {
    h = hidden[j];
    sum = 0.0;
    for (k = 1; k <= no; k++) {
      sum += delta_o[k] * who[j*(no+1)+k];
    }
    delta_h[j] = h * (1.0 - h) * sum;
    errsum += ABS(delta_h[j]);
  }
  *err = errsum;
}


void bpnn_adjust_weights(float *delta,int ndelta,float* ly,int nly,float* w,float* oldw)
{
  float new_dw;
  int k, j;
  ly[0] = 1.0;
  //eta = 0.3;
  //momentum = 0.3;
#ifdef OPEN
  omp_set_num_threads(NUM_THREAD);
  #pragma omp parallel for  \
      shared(oldw, w, delta) \
          private(j, k, new_dw) \
          firstprivate(ndelta, nly, momentum) 
#endif
  for (j = 1; j <= ndelta; j++) {
    for (k = 0; k <= nly; k++) {
      new_dw = ((ETA * delta[j] * ly[k]) + (MOMENTUM * oldw[k*(ndelta+1)+j]));
          w[k*(ndelta+1)+j] += new_dw;
          oldw[k*(ndelta+1)+j] = new_dw;
    }
  }
}


void bpnn_feedforward(BPNN *net)
{
  int in, hid, out;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;

  /*** Feed forward input activations. ***/
  bpnn_layerforward(net->input_units, net->hidden_units,
      net->input_weights, in, hid);
  bpnn_layerforward(net->hidden_units, net->output_units,
      net->hidden_weights, hid, out);

}


void bpnn_train(BPNN *net,float* eo,float* eh)
{
  int in, hid, out;
  float out_err, hid_err;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;

  /*** Feed forward input activations. ***/
  bpnn_layerforward(net->input_units, net->hidden_units,
      net->input_weights, in, hid);
  bpnn_layerforward(net->hidden_units, net->output_units,
      net->hidden_weights, hid, out);
  /*** Compute error on output and hidden units. ***/
  bpnn_output_error(net->output_delta, net->target, net->output_units,
      out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out,
      net->hidden_weights, net->hidden_units, &hid_err);
  *eo = out_err;
  *eh = hid_err;

  /*** Adjust input and hidden weights. ***/
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid,
      net->hidden_weights, net->hidden_prev_weights);
  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in,
      net->input_weights, net->input_prev_weights);

}




void bpnn_save(BPNN *net,char *filename)
{
  int n1, n2, n3, i, j, memcnt;
  float dvalue, *w;
  char *mem;
  ///add//
  FILE *pFile;
  pFile = fopen( filename, "w+" );
  ///////
  /*
  if ((fd = creat(filename, 0644)) == -1) {
    printf("BPNN_SAVE: Cannot create '%s'\n", filename);
    return;
  }
  */

  n1 = net->input_n;  n2 = net->hidden_n;  n3 = net->output_n;
  printf("Saving %dx%dx%d network to '%s'\n", n1, n2, n3, filename);
  //fflush(stdout);

  //write(fd, (char *) &n1, sizeof(int));
  //write(fd, (char *) &n2, sizeof(int));
  //write(fd, (char *) &n3, sizeof(int));

  fwrite( (char *) &n1 , sizeof(char), sizeof(char), pFile);
  fwrite( (char *) &n2 , sizeof(char), sizeof(char), pFile);
  fwrite( (char *) &n3 , sizeof(char), sizeof(char), pFile);



  memcnt = 0;
  w = net->input_weights;
  mem = (char *) malloc ((unsigned) ((n1+1) * (n2+1) * sizeof(float)));
  for (i = 0; i <= n1; i++) {
    for (j = 0; j <= n2; j++) {
      dvalue = w[i*(n2+1)+j];
      fastcopy(&mem[memcnt], &dvalue, sizeof(float));
      memcnt += sizeof(float);
    }
  }
  //write(fd, mem, (n1+1) * (n2+1) * sizeof(float));
  fwrite( mem , (unsigned)(sizeof(float)), (unsigned) ((n1+1) * (n2+1) * sizeof(float)) , pFile);
  free(mem);

  memcnt = 0;
  w = net->hidden_weights;
  mem = (char *) malloc ((unsigned) ((n2+1) * (n3+1) * sizeof(float)));
  for (i = 0; i <= n2; i++) {
    for (j = 0; j <= n3; j++) {
      dvalue = w[i*(n3+1)+j];
      fastcopy(&mem[memcnt], &dvalue, sizeof(float));
      memcnt += sizeof(float);
    }
  }
  //write(fd, mem, (n2+1) * (n3+1) * sizeof(float));
  fwrite( mem , sizeof(float), (unsigned) ((n2+1) * (n3+1) * sizeof(float)) , pFile);
  free(mem);

  fclose(pFile);
  return;
}


BPNN *bpnn_read(char* filename)
{
  char *mem;
  BPNN *new_bpnn;
  int fd, n1, n2, n3, i, j, memcnt;

  if ((fd = open(filename, 0, 0644)) == -1) {
    return (NULL);
  }

  printf("Reading '%s'\n", filename);  //fflush(stdout);

  read(fd, (char *) &n1, sizeof(int));
  read(fd, (char *) &n2, sizeof(int));
  read(fd, (char *) &n3, sizeof(int));
  new_bpnn = bpnn_internal_create(n1, n2, n3);

  printf("'%s' contains a %dx%dx%d network\n", filename, n1, n2, n3);
  printf("Reading input weights...");  //fflush(stdout);

  memcnt = 0;
  mem = (char *) malloc ((unsigned) ((n1+1) * (n2+1) * sizeof(float)));
  read(fd, mem, (n1+1) * (n2+1) * sizeof(float));
  for (i = 0; i <= n1; i++) {
    for (j = 0; j <= n2; j++) {
      fastcopy(&(new_bpnn->input_weights[i*(n2+1)+j]), &mem[memcnt], sizeof(float));
      memcnt += sizeof(float);
    }
  }
  free(mem);
  printf("Done\nReading hidden weights...");  //fflush(stdout);

  memcnt = 0;
  mem = (char *) malloc ((unsigned) ((n2+1) * (n3+1) * sizeof(float)));
  read(fd, mem, (n2+1) * (n3+1) * sizeof(float));
  for (i = 0; i <= n2; i++) {
    for (j = 0; j <= n3; j++) {
      fastcopy(&(new_bpnn->hidden_weights[i*(n3+1)+j]), &mem[memcnt], sizeof(float));
      memcnt += sizeof(float);
    }
  }
  free(mem);
  close(fd);

  printf("Done\n");  //fflush(stdout);

  bpnn_zero_weights(new_bpnn->input_prev_weights, n1, n2);
  bpnn_zero_weights(new_bpnn->hidden_prev_weights, n2, n3);

  return (new_bpnn);
}

void load(BPNN *net)
{
  float *units;
  int nr, nc, imgsize, i, j, k;

  nr = layer_size;
  
  imgsize = nr * nc;
  units = net->input_units;

  k = 1;
  for (i = 0; i < nr; i++) {
	  units[k] = (float) rand()/RAND_MAX ;
	  k++;
    }
}

int setup(int argc, char *argv[])
{
	
  int seed;

  if (argc!=2){
  fprintf(stderr, "usage: backprop <num of input elements>\n");
  exit(0);
  }
  layer_size = atoi(argv[1]);
  if (layer_size%16!=0){
  fprintf(stderr, "The number of input points must be divided by 16\n");
  exit(0);
  }
  

  seed = 7;   
  printf("Random number generator seed: %d\n", seed);
  srand(seed);

  BPNN *net;
  int i;
  float out_err, hid_err;
  net = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)
  
  printf("Input layer size : %d\n", layer_size);
  load(net);
  //entering the training kernel, only one iteration
  printf("Starting training kernel\n");
  bpnn_train_cuda(net, &out_err, &hid_err);
  bpnn_free(net);
  printf("Training done\n");

  exit(0);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
	setup(argc, argv);
}


void bpnn_train_cuda(BPNN *net, float *eo, float *eh)
{
  int in, hid, out;
  float out_err, hid_err;
  
  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;   
   
#ifdef GPU  
  float *output_hidden_cuda;
  float *hidden_partial_sum;
  float sum;
  num_blocks = in / 16;  
  dim3  grid( 1 , num_blocks);
  dim3  threads(16 , 16);
  
  cudaMallocManaged((void**) &output_hidden_cuda, (hid + 1) * sizeof(float));
  cudaMallocManaged((void**) &hidden_partial_sum, num_blocks * WIDTH * sizeof(float));
  
#endif

#ifdef PREF
  // Prefetch the data to the GPU
  int device = -1;
  cudaGetDevice(&device);

  cudaStream_t stream1;
  cudaStreamCreate(&stream1);

  cudaStream_t stream2;
  cudaStreamCreate(&stream2);

  cudaStream_t stream3;
  cudaStreamCreate(&stream3);

  cudaStream_t stream4;
  cudaStreamCreate(&stream4);

  cudaStream_t stream5;
  cudaStreamCreate(&stream5);

  cudaStream_t stream6;
  cudaStreamCreate(&stream6);

  cudaMemPrefetchAsync(net->input_units, (in + 1) * sizeof(float), device, stream1);
  cudaMemPrefetchAsync(net->input_weights,(in + 1) * (hid + 1) * sizeof(float), device, stream2);
#endif

#ifdef CPU

  printf("Performing CPU computation\n");
  bpnn_layerforward(net->input_units, net->hidden_units,net->input_weights, in, hid);

#endif

#ifdef GPU
 
  printf("Performing GPU computation\n");
  
  //printf("in= %d, hid = %d, numblocks = %d\n", in, hid, num_blocks);
  

#ifdef PREF
  bpnn_layerforward_CUDA<<< grid, threads, 0, stream2 >>>(net->input_units,
					      output_hidden_cuda,
					      net->input_weights,
					      hidden_partial_sum,
					      in,
					      hid);
#else  
  bpnn_layerforward_CUDA<<< grid, threads >>>(net->input_units,
					      output_hidden_cuda,
					      net->input_weights,
					      hidden_partial_sum,
					      in,
					      hid);
#endif


  cudaThreadSynchronize();
  
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("bpnn kernel error: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
  
     
  for (int j = 1; j <= hid; j++) {
    sum = 0.0;
    for (int k = 0; k < num_blocks; k++) {	
      sum += hidden_partial_sum[k * hid + j-1] ;
    }
    sum += net->input_weights[j];
    net-> hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
  }
  #endif

  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);

#ifdef CPU

  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in, net->input_weights, net->input_prev_weights);

#endif  

#ifdef PREF
  // Prefetch the data to the GPU
  cudaGetDevice(&device);
  cudaMemPrefetchAsync(net->hidden_delta, (hid + 1) * sizeof(float), device, stream3);
  cudaMemPrefetchAsync(net->input_prev_weights,(in + 1) * (hid + 1) * sizeof(float), device, stream4);
  cudaMemPrefetchAsync(net->input_weights2, (in + 1) * (hid + 1) * sizeof(float), device, stream5);
#endif


#ifdef GPU

#ifdef PREF
  bpnn_adjust_weights_cuda<<< grid, threads, 0, stream6>>>(net->hidden_delta,  
						hid, 
						net->input_units, 
						in,
						net->input_weights2, 
						net->input_prev_weights
						);
#else
  bpnn_adjust_weights_cuda<<< grid, threads >>>(net->hidden_delta,  
						hid, 
						net->input_units, 
						in,
						net->input_weights2, 
						net->input_prev_weights
						);
#endif

  cudaDeviceSynchronize();

#define DEBUG
#ifdef DEBUG
  FILE *fp2 = fopen("result.txt","w");
  fprintf(fp2,"Input_units:\n");
  for(int i = 0; i < in + 1; i ++)
    fprintf(fp2,"%f ", net->input_units[i]);
  fprintf(fp2,"\n");
  fprintf(fp2,"Input_weight_one_dim:\n");
  for(int i = 0; i < in + 1; i ++){
    for(int j = 0; j < hid + 1; j++)
      fprintf(fp2,"%f ", net->input_weights2[i*(hid+1)+j]);
    fprintf(fp2,"\n");
  }
  fclose(fp2);
  
#endif

  cudaFree(output_hidden_cuda);
  cudaFree(hidden_partial_sum);
#endif   
  
  
  

}
