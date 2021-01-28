/***********************************************************************************
  Implementing Breadth first search on CUDA using algorithm given in HiPC'07
  paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

  Copyright (c) 2008 International Institute of Information Technology - Hyderabad. 
  All rights reserved.

  Permission to use, copy, modify and distribute this software and its documentation for 
  educational purpose is hereby granted without fee, provided that the above copyright 
  notice and this permission notice appear in all copies of this software and that you do 
  not sell the software.

  THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR 
  OTHERWISE.

  Created by Pawan Harish.
 ************************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

#define MAX_THREADS_PER_BLOCK 512

int no_of_nodes;
int edge_list_size;
FILE *fp;

//Structure to hold a node information
struct Node
{
	int starting;
	int no_of_edges;
};

#include "kernel.cu"
#include "kernel2.cu"

void BFSGraph(int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	no_of_nodes=0;
	edge_list_size=0;
	BFSGraph( argc, argv);
}

void Usage(int argc, char**argv){

fprintf(stderr,"Usage: %s <input_file>\n", argv[0]);

}
////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph( int argc, char** argv) 
{

    char *input_f;
	if(argc!=2){
	Usage(argc, argv);
	exit(0);
	}
	
	input_f = argv[1];
	printf("Reading File\n");
	//Read in Graph from a file
	fp = fopen(input_f,"r");
	if(!fp)
	{
		printf("Error Reading graph file\n");
		return;
	}

	int source = 0;

	fscanf(fp,"%d",&no_of_nodes);

	int num_of_blocks = 1;
	int num_of_threads_per_block = no_of_nodes;

	//Make execution Parameters according to the number of nodes
	//Distribute threads across multiple Blocks if necessary
	if(no_of_nodes>MAX_THREADS_PER_BLOCK)
	{
		num_of_blocks = (int)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK); 
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}


	//Allocate the Node list
	Node* graph_nodes;
	cudaMallocManaged(  &graph_nodes, sizeof(Node)*no_of_nodes) ;

	//Allocate the Mask
	bool* graph_mask;
	cudaMallocManaged( &graph_mask, sizeof(bool)*no_of_nodes) ;

	bool* updating_graph_mask;
	cudaMallocManaged( &updating_graph_mask, sizeof(bool)*no_of_nodes) ;

	//Allocate the Visited nodes array
	bool* graph_visited;
	cudaMallocManaged( &graph_visited, sizeof(bool)*no_of_nodes) ;

	int start, edgeno;   
	// initalize the memory
	for( unsigned int i = 0; i < no_of_nodes; i++) 
	{
                fscanf(fp,"%d %d",&start,&edgeno);
		graph_nodes[i].starting = start;
		graph_nodes[i].no_of_edges = edgeno;
		graph_mask[i]=false;
		updating_graph_mask[i]=false;
		graph_visited[i]=false;
	}

	//read the source node from the file
	fscanf(fp,"%d",&source);
	source=0;

	//set the source node as true in the mask
	graph_mask[source]=true;
	graph_visited[source]=true;

	fscanf(fp,"%d",&edge_list_size);

	//Allocate the Edge List
	int* graph_edges;
	cudaMallocManaged( &graph_edges, sizeof(int)*edge_list_size) ;

	int id,edgeCost;
	for(int i=0; i < edge_list_size ; i++)
	{
		fscanf(fp,"%d",&id);
		fscanf(fp,"%d",&edgeCost);
		graph_edges[i] = id;
	}

	if(fp)
		fclose(fp);    

	printf("Read File\n");

	// allocate mem for the result
	int* cost;
	cudaMallocManaged( (void**) &cost, sizeof(int)*no_of_nodes);

	for(int i=0;i<no_of_nodes;i++)
		cost[i]=-1;
	cost[source]=0;
	
        //make a bool to check if the execution is over
        bool *d_over;
        cudaMalloc( (void**) &d_over, sizeof(bool));

	printf("Copied Everything to GPU memory\n");

#ifdef PREF
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

	cudaStream_t stream7;
	cudaStreamCreate(&stream7);

	cudaMemPrefetchAsync( graph_nodes, sizeof(Node)*no_of_nodes, device, stream1);
	cudaMemPrefetchAsync( graph_edges, sizeof(int)*edge_list_size, device, stream2);
	cudaMemPrefetchAsync( graph_mask, sizeof(bool)*no_of_nodes, device, stream3);
	cudaMemPrefetchAsync( updating_graph_mask, sizeof(bool)*no_of_nodes, device, stream4);
	cudaMemPrefetchAsync( graph_visited, sizeof(bool)*no_of_nodes, device, stream5);
	cudaMemPrefetchAsync( cost, sizeof(int)*no_of_nodes, device, stream6);
#endif

	// setup execution parameters
	dim3  grid( num_of_blocks, 1, 1);
	dim3  threads( num_of_threads_per_block, 1, 1);

	int k=0;
	printf("Start traversing the tree\n");
        bool stop;	
	//Call the Kernel untill all the elements of Frontier are not false
	do
	{
                //if no thread changes this value then the loop stops
                stop=false;
                cudaMemcpy( d_over, &stop, sizeof(bool), cudaMemcpyHostToDevice) ;

#ifdef PREF
		Kernel<<< grid, threads, 0, stream7>>>( graph_nodes, graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes);
		// check if kernel execution generated and error
		

		Kernel2<<< grid, threads, 0, stream7>>>( graph_mask, updating_graph_mask, graph_visited, d_over, no_of_nodes);
		// check if kernel execution generated and error
#else
		Kernel<<< grid, threads, 0 >>>( graph_nodes, graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes);
		// check if kernel execution generated and error
		

		Kernel2<<< grid, threads, 0 >>>( graph_mask, updating_graph_mask, graph_visited, d_over, no_of_nodes);
		// check if kernel execution generated and error
		
#endif		

                cudaMemcpy( &stop, d_over, sizeof(bool), cudaMemcpyDeviceToHost) ;
		k++;
	}
	while(stop); //if no thread changes this value then the loop stops


        cudaDeviceSynchronize();

	printf("Kernel Executed %d times\n",k);

	//Store the result into a file
	FILE *fpo = fopen("result.txt","w");
	for(int i=0;i<no_of_nodes;i++)
		fprintf(fpo,"%d) cost:%d\n",i,cost[i]);
	fclose(fpo);
	printf("Result stored in result.txt\n");


	// cleanup memory
	cudaFree(graph_nodes);
	cudaFree(graph_edges);
	cudaFree(graph_mask);
	cudaFree(updating_graph_mask);
	cudaFree(graph_visited);
	cudaFree(cost);
	cudaFree(d_over);
}
