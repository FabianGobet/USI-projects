// #include <omp.h>
#include "walltime.h"
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <unistd.h>

#define NUM_ITERATIONS 100

// Example benchmarks
// 0.008s ~0.8MB
//#define N 100000
// 0.1s ~8MB
// #define N 1000000
// 1.1s ~80MB
// #define N 10000000
// 13s ~800MB
// #define N 100000000
// 127s 16GB
//#define N 1000000000
#define EPSILON 0.1

using namespace std;

int main(int argc, char* argv[]) {
  double time_serial, time_start = 0.0;
  double *a, *b;
  int p = 1;
  int N = 100000;

  if(argc == 3){
    p = atoi(argv[1]);
    N = atoi(argv[2]);
  }
  cout << "Numbers of threads:" << p << "\nVector length: " << N << endl;

  // Allocate memory for the vectors as 1-D arrays
  a = new double[N];
  b = new double[N];

  // Initialize the vectors with some values
  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = i / 10.0;
  }

  long double alpha = 0;
  // serial execution
  // Note that we do extra iterations to reduce relative timing overhead
  time_start = wall_time();
  for (int iterations = 0; iterations < NUM_ITERATIONS; iterations++) {
    alpha = 0.0;
    for (int i = 0; i < N; i++) {
      alpha += a[i] * b[i];
    }
  }
  time_serial = wall_time() - time_start;
  cout << "Serial execution time = " << time_serial << " sec" << endl;

  long double alpha_parallel = 0;
  double time_red = 0;
  double time_critical = 0;

  //   TODO: Write parallel version (2 ways!)

    //   i.  Using reduction pragma
  time_start = wall_time();
  for (int iterations = 0; iterations < NUM_ITERATIONS; iterations++) {
    alpha_parallel = 0.0;
    #pragma omp parallel for default(none) shared(a,b,N) reduction(+:alpha_parallel) num_threads(p)
    for (int i = 0; i < N; i++) {
      alpha_parallel += a[i] * b[i];
    }
  }
  time_red = wall_time() - time_start;

  alpha_parallel = 0;
  //   ii. Using  critical pragma
  time_start = wall_time();
  for (int iterations = 0; iterations < NUM_ITERATIONS; iterations++) {
    alpha_parallel = 0.0;
    #pragma omp parallel for shared(a,b,N) num_threads(p)
    for (int i = 0; i < N; i++) {
      #pragma omp critical
      alpha_parallel += a[i] * b[i];
    }
  }
  time_critical = wall_time() - time_start;
  
  


  if ((fabs(alpha_parallel - alpha) / fabs(alpha_parallel)) > EPSILON) {
    cout << "parallel reduction: " << alpha_parallel << ", serial: " << alpha
         << "\n";
    cerr << "Alpha not yet implemented correctly!\n";
    exit(1);
  }
  cout << "Parallel dot product = " << alpha_parallel
       << "\n time using reduction method = " << time_red
       << "\n sec, time using critical method " << time_critical << " sec\n"
       << endl;

  // De-allocate memory
  delete[] a;
  delete[] b;

  return 0;
}
