#include "walltime.h"
#include <math.h>
#include <stdlib.h>

#include <omp.h>

int main(int argc, char *argv[]) {
  int N = 2000000000;
  double up = 1.00000001;
  double Sn = 1.00000001;
  int n;
  /* allocate memory for the recursion */
  double *opt = (double *)malloc((N + 1) * sizeof(double));

  if (opt == NULL)
    die("failed to allocate problem size");

  double time_start = wall_time();
  // TODO: YOU NEED TO PARALLELIZE THIS LOOP
  
  int div = 100;
  int tnum = 7;

  /*
  if(argc==3) {
    div = atoi(argv[1]);
    tnum = atoi(argv[2]);
  }
  */

  long long int blocksize = (long long int)N/div;
  int mini = (blocksize < N+1 ? blocksize : N+1);

  double up2 = 1;

  for( int i=0; i < mini; i++){
    opt[i] = Sn;
    Sn *= up;
    up2 *= up;
  }
  
  long long int i, j;

  for(i=blocksize; i<N+1; i+=blocksize){
    #pragma omp parallel for shared(i,blocksize,opt,N) num_threads(tnum)
    for(j=i; j<(i+blocksize>N+1 ? N+1 : i+blocksize); j++)
      opt[j] = opt[j-blocksize]*up2;
  }
  Sn = opt[N];
  

  /*
  #pragma omp parallel for num_threads(tnum) shared(blocksize,N,opt,up2)
  for(int i = 0; i<tnum; i++)
    for(int j = mini+i; j <=N; j+=mini){
      opt[j] = opt[j-mini]*up2;
    }
  Sn = opt[N];
  */

  //printf("%i Threads, %i division factor:\n", tnum, div);
  printf("Parallel RunTime   :  %f seconds\n", wall_time() - time_start);
  printf("Final Result Sn    :  %.17g \n", Sn);

  
  double temp = 0.0;
  for (n = 0; n <= N; ++n) {
    //printf("%i ", opt[n]);
    temp += opt[n] * opt[n];
  }
  printf("Result ||opt||^2_2 :  %f\n", temp / (double)N);
  printf("\n");
  

  free(opt);

  return 0;
}
