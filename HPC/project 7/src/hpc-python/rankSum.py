from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# LOWER CASE
# total_sum = comm.allreduce(rank, op=MPI.SUM) # all ranks have the sum value
total_sum = comm.reduce(rank, op=MPI.SUM, root=0) # only rank 0 gets the sum value

if rank == 0:
    print(f"Total Sum (Pickle-Based): {total_sum}")


# UPPER CASE
rank_array = np.array(rank, dtype='i')

# sum_array = np.empty(1, dtype='i') # all ranks have the sum value

if rank == 0: # only rank 0 gets the sum value
    sum_array = np.empty(1, dtype='i')
else:
    sum_array = None

#comm.Allreduce(rank_array, sum_array, op=MPI.SUM)
comm.Reduce(rank_array, sum_array, op=MPI.SUM, root=0)

if rank == 0:
    print(f"Total Sum (Array-Based): {total_sum}")