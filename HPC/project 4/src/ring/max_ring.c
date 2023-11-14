/****************************************************************
 *                                                              *
 * This file has been written as a sample solution to an        *
 * exercise in a course given at the CSCS-USI Summer School.    *
 * It is made freely available with the understanding that      *
 * every copy of this file must include this header and that    *
 * CSCS/USI take no responsibility for the use of the enclosed  *
 * teaching material.                                           *
 *                                                              *
 * Purpose: Parallel maximum using a ping-pong                      *
 *                                                              *
 * Contents: C-Source                                           *
 *                                                              *
 ****************************************************************/


#include <stdio.h>
#include <mpi.h>


int main (int argc, char *argv[])
{
    int my_rank, size;
    int snd_buf, rcv_buf;
    int right, left;
    int max, i;

    MPI_Status  status;
    MPI_Request request;


    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    MPI_Comm_size(MPI_COMM_WORLD, &size);


    right = (my_rank+1)%(size);/* get rank of neighbor to your right */
    left  = (my_rank-1)%(size);/* get rank of neighbor to your left */
    
    snd_buf = (3*my_rank)%(2*size);
    max = snd_buf;

    for(int i = 0; i<size-1; i++){
        MPI_Send(&snd_buf,1,MPI_INT,right,0,MPI_COMM_WORLD);
        MPI_Recv(&rcv_buf,1,MPI_INT,left,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        if(rcv_buf>max)
            max=rcv_buf;
        snd_buf = rcv_buf;
    }

    printf ("Process %i:\tMax = %i\n", my_rank, max);

    MPI_Finalize();
    return 0;
}
