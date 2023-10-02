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
  const int block_size = 64;

  for (int kk = 0; kk < n; kk += block_size) {
        for (int jj = 0; jj < n; jj += block_size) {

            // upper bounds for blocks. 
            int k_max = kk + block_size > n ? n : kk + block_size; // max block size; clamps if over n
            int j_max = jj + block_size > n ? n : jj + block_size;

            for (int i = 0; i < n; i++) { // iterate through each row of A
                for (int k = kk; k < k_max; k++) { // iterate through rows of current block of A

                    double r = A[i + k * n]; // prefetch value to reduce memory access

                    for (int j = jj; j < j_max; j++) { // iterate through columns of current block of B
                        C[i + j * n] += r * B[k + j * n];
                    }
                }
            }
        }
    }
}
