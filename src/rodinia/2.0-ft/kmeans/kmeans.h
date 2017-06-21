#ifndef _H_FUZZY_KMEANS
#define _H_FUZZY_KMEANS

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

/* rmse.c */
float   euclid_dist_2        (float*, float*, int);
int     find_nearest_point   (float* , int, float**, int);
float	rms_err(float**, int, int, float**, int);

/* cluster.c */
int     cluster(int, int, float**, int, int, float, int*, float***, float*, int, int);

/* kmeans_clustering.c */
float **kmeans_clustering(float**, int, int, int, float, int*);

#endif
