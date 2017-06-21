/*************************************************************************/
/**   File:         cluster.c                                           **/
/**   Description:  Takes as input a file, containing 1 data point per  **/
/**                 per line, and performs a fuzzy c-means clustering   **/
/**                 on the data. Fuzzy clustering is performed using    **/
/**                 min to max clusters and the clustering that gets    **/
/**                 the best score according to a compactness and       **/
/**                 separation criterion are returned.                  **/
/**   Author:  Brendan McCane                                           **/
/**            James Cook University of North Queensland.               **/
/**            Australia. email: mccane@cs.jcu.edu.au                   **/
/**                                                                     **/
/**   Edited by: Jay Pisharath, Wei-keng Liao                           **/
/**              Northwestern University.                               **/
/**																		**/
/**   ================================================================  **/
/**																		**/
/**   Edited by: Shuai Che, David Tarjan, Sang-Ha Lee					**/
/**				 University of Virginia									**/
/**																		**/
/**   Description:	No longer supports fuzzy c-means clustering;	 	**/
/**					only regular k-means clustering.					**/
/**					No longer performs "validity" function to analyze	**/
/**					compactness and separation crietria; instead		**/
/**					calculate root mean squared error.					**/
/**                                                                     **/
/*************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <float.h>

#include "kmeans.h"

extern double wtime(void);
float	min_rmse_ref = FLT_MAX;			/* reference min_rmse value */

/*---< cluster() >-----------------------------------------------------------*/
int cluster(int      npoints,				/* number of data points */
            int      nfeatures,				/* number of attributes for each point */
            float  **features,			/* array: [npoints][nfeatures] */                  
            int      min_nclusters,			/* range of min to max number of clusters */
			int		 max_nclusters,
            float    threshold,				/* loop terminating factor */
            int     *best_nclusters,		/* out: number between min and max with lowest RMSE */
            float ***cluster_centres,		/* out: [best_nclusters][nfeatures] */
			float	*min_rmse,				/* out: minimum RMSE */
			int		 isRMSE,				/* calculate RMSE */
			int		 nloops					/* number of iteration for each number of clusters */
			)
{    
	int		nclusters;						/* number of clusters k */	
	int		index =0;						/* number of iteration to reach the best RMSE */
	int		rmse;							/* RMSE for each clustering */
    int    *membership;						/* which cluster a data point belongs to */
    float **tmp_cluster_centres;			/* hold coordinates of cluster centers */
	int		i;

	/* allocate memory for membership */
    membership = (int*) malloc(npoints * sizeof(int));

	/* sweep k from min to max_nclusters to find the best number of clusters */
	for(nclusters = min_nclusters; nclusters <= max_nclusters; nclusters++)
	{
		if (nclusters > npoints) break;	/* cannot have more clusters than points */

		/* allocate device memory, invert data array (@ kmeans_cuda.cu) */
		allocateMemory(npoints, nfeatures, nclusters, features);

		/* iterate nloops times for each number of clusters */
		for(i = 0; i < nloops; i++)
		{
			/* initialize initial cluster centers, CUDA calls (@ kmeans_cuda.cu) */
			tmp_cluster_centres = kmeans_clustering(features,
													nfeatures,
													npoints,
													nclusters,
													threshold,
													membership);

			if (*cluster_centres) {
				free((*cluster_centres)[0]);
				free(*cluster_centres);
			}
			*cluster_centres = tmp_cluster_centres;
	        
					
			/* find the number of clusters with the best RMSE */
			if(isRMSE)
			{
				rmse = rms_err(features,
							   nfeatures,
							   npoints,
							   tmp_cluster_centres,
							   nclusters);
				
				if(rmse < min_rmse_ref){
					min_rmse_ref = rmse;			//update reference min RMSE
					*min_rmse = min_rmse_ref;		//update return min RMSE
					*best_nclusters = nclusters;	//update optimum number of clusters
					index = i;						//update number of iteration to reach best RMSE
				}
			}			
		}
		
		deallocateMemory();							/* free device memory (@ kmeans_cuda.cu) */
	}

    free(membership);

    return index;
}

