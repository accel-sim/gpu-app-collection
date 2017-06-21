

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "backprop.h"
//#include "omp.h"

extern char *strcpy();
extern void exit();

int layer_size = 0;

backprop_face()
{
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
}
const char* goldfile;
int setup(argc, argv)
int argc;
char *argv[];
{
	
  int seed;

  if (argc<2){
  fprintf(stderr, "usage: backprop <num of input elements>\n");
  exit(0);
  }
  layer_size = atoi(argv[1]);
  if (layer_size%16!=0){
  fprintf(stderr, "The number of input points must be divided by 16\n");
  exit(0);
  }
  if(argc>=3)
	  goldfile = argv[2];

  seed = 7;   
  bpnn_initialize(seed);
  backprop_face();

  exit(0);
}
