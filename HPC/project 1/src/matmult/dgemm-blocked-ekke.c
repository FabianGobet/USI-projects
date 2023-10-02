/* 
    Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER= gnu

    Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines
 
CC = cc
OPT = -O3
CFLAGS = -Wall -std=gnu99 $(OPT)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

*/
#include <stdlib.h>
const char* dgemm_desc = "Naive, three-loop dgemm.";

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */    
void square_dgemm (int n, double* A, double* B, double* C)
{
  // TODO: Implement the blocking optimization
  const int block_size = 32;

  for (int kk = 0; kk < n; kk += block_size) {
        for (int jj = 0; jj < n; jj += block_size) {

            // Compute upper bounds for blocks. 
            int k_max = kk + block_size > n ? n : kk + block_size;
            int j_max = jj + block_size > n ? n : jj + block_size;

            for (int i = 0; i < n; i++) {
                for (int k = kk; k < k_max; k++) {

                    double r = A[i + k * n];  // Prefetch value from A

                    for (int j = jj; j < j_max; j++) {
                        C[i + j * n] += r * B[k + j * n];
                    }
                }
            }
        }
    }
}
