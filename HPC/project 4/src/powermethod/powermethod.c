/****************************************************************
 *                                                              *
 * This file has been written as a sample solution to an        *
 * exercise in a course given at the CSCS-USI Summer School.    *
 * It is made freely available with the understanding that      *
 * every copy of this file must include this header and that    *
 * CSCS/USI take no responsibility for the use of the enclosed  *
 * teaching material.                                           *
 *                                                              *
 * Purpose: : Parallel matrix-vector multiplication and the     *
 *            and power method                                  *
 *                                                              *
 * Contents: C-Source                                           *
 *                                                              *
 ****************************************************************/


#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "hpc-power.c"

/*
void printIt(double* toprint, int size, int rank, char* str){
    printf("Rank %d, %s: ",rank,str);
    for(int i=0; i<size; i++)
        printf("%f ",toprint[i]);
    printf("\n");
}
*/

double norm(double *vector, int n){
    double sum = 0.0;
    for(int i=0; i<n; i++)
        sum += vector[i]*vector[i];
    return sqrt(sum);
}

void matvec(double* A, int n, int numrows, double* vector, double* result){
    for(int i=0; i<numrows; i++){
        double innerproduct = 0.0;
        for(int j=0; j<n; j++)
            innerproduct += A[i*n+j]*vector[j];
        result[i]=innerproduct;
    }
}

double* randVector(int size, double minRange, double maxRange){
    srand((unsigned)time(NULL));
    double* vector = (double *)(malloc(size*sizeof(double)));
        for(int i = 0; i<size; i++)
            vector[i] = minRange + ((double)rand() / RAND_MAX) * (maxRange - minRange);
    return vector;
}

void powerMethod(double* A, double* vector, double* result ,int my_rank, int n, int numrows, MPI_Comm comm){
    if(my_rank==0){
        double nrm = norm(vector,n);
        for(int i=0; i<n; i++)
            vector[i] = vector[i]/nrm;
    }
    MPI_Bcast(vector, n, MPI_DOUBLE, 0, comm);
    matvec(A, n, numrows, vector, result);
    if(my_rank!=0)
        MPI_Gather(result, numrows, MPI_DOUBLE, NULL, 0, MPI_DOUBLE, 0, comm);
    else 
        MPI_Gather(result, numrows, MPI_DOUBLE, vector, numrows, MPI_DOUBLE, 0, comm);
}



int main (int argc, char *argv[])
{
    int my_rank, size;
    int n = 9600;
    double start, end;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(argc>1) 
        n = atoi(argv[1]);

    if(n%size!=0){
        printf("please run this with 'n' divisible by number of processors\n");
        MPI_Finalize();
        exit(1);
    }

    int numrows = n/size;
    double* A = hpc_generateMatrix(n, numrows);
    double* vector = (double *)(malloc(n*sizeof(double)));
    double* result = (double *)(malloc(numrows*sizeof(double)));

    if(my_rank==0){
        vector = randVector(n, -5, 5);
        start = hpc_timer();
    }

    for(int i=0; i<100; i++){
        powerMethod(A, vector, result , my_rank, n, numrows, MPI_COMM_WORLD);
    }

    if(my_rank==0) {
        end = hpc_timer() - start;
        int correct = hpc_verify(vector, n, end);
        //printf("Result %s, in %f seconds.\n", correct==0 ? "incorrect":"correct",end);
        printf("%d,%d,%f\n", n, size, end);
    }
    
    free(A);
    free(result);
    free(vector);

    MPI_Finalize();
    return 0;
}
