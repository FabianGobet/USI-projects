#include "walltime.h"
#include <iostream>
#include <random>
#include <omp.h>

#define VEC_SIZE 1000000000
#define BINS 16

using namespace std;

int main() {
  double time_start, time_end;

  // Initialize random number generator
  unsigned int seed = 123;
  float mean = BINS / 2.0;
  float sigma = BINS / 12.0;
  std::default_random_engine generator(seed);
  std::normal_distribution<float> distribution(mean, sigma);

  // Generate random sequence
  // Note: normal distribution is on interval [-inf; inf]
  //       we want [0; BINS-1]
  int *vec = new int[VEC_SIZE];
  //#pragma omp parallel parallel for shared(vec) private(i)
  for (long i = 0; i < VEC_SIZE; ++i) {
    vec[i] = int(distribution(generator));
    if (vec[i] < 0)
      vec[i] = 0;
    if (vec[i] > BINS - 1)
      vec[i] = BINS - 1;
  }

  //#pragma omp barrier
  // Initialize histogram
  // Set all bins to zero
  long dist[BINS];
  //#pragma omp parallel parallel for shared(dist) private(i)
  for (int i = 0; i < BINS; ++i) {
    dist[i] = 0;
  }

  time_start = wall_time();

  /*
  // TODO Parallelize the histogram computation
  #pragma omp parallel for shared(dist,vec)
  for (long i = 0; i < VEC_SIZE; ++i) {
    #pragma omp atomic
    dist[vec[i]]++;
  }
  */

 
  // TODO Parallelize the histogram computation
  #pragma omp parallel 
  {
    long local_dist[BINS] = {0};

  #pragma omp for
    for (long i = 0; i < VEC_SIZE; ++i) {
        local_dist[vec[i]]++;
    }
    
    for(int i=0; i<BINS; i++){
      #pragma omp atomic
      dist[i] += local_dist[i];
    }
  #pragma barrier
  }
 

  time_end = wall_time();

  // Write results
  for (int i = 0; i < BINS; ++i) {
    cout << "dist[" << i << "]=" << dist[i] << endl;
  }
  cout << "Time: " << time_end - time_start << " sec" << endl;

  delete[] vec;

  return 0;
}
