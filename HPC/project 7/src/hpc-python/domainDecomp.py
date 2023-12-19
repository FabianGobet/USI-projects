from mpi4py import MPI
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Compute the dimensions for the Cartesian grid
dims = MPI.Compute_dims(size, [0, 0])
periods = [False, False]  # Non-periodic in both dimensions

# Create the Cartesian topology
cart_comm = comm.Create_cart(dims, periods=periods, reorder=True)
coords = cart_comm.Get_coords(rank)

# Determine the neighboring processes
east, west = cart_comm.Shift(0, 1)  # Shift along the first dimension (x-axis)
north, south = cart_comm.Shift(1, 1) # Shift along the second dimension (y-axis)

# Initialize variables for neighbor ranks
neighbor_ranks = {'east': None, 'west': None, 'north': None, 'south': None}

# Function to exchange rank with a neighbor
def exchange_rank(neighbor, direction):
    if neighbor != MPI.PROC_NULL:
        send_rank = np.array(rank, dtype='i')
        recv_rank = np.empty(1, dtype='i')
        cart_comm.Sendrecv(sendbuf=send_rank, dest=neighbor, recvbuf=recv_rank, source=neighbor)
        neighbor_ranks[direction] = recv_rank[0]

# Exchange ranks with neighbors
exchange_rank(east, 'east')
exchange_rank(west, 'west')
exchange_rank(north, 'north')
exchange_rank(south, 'south')

# Output the results
print(f"Rank: {rank}, Coordinates: {coords}, Neighbors - East: {east}, West: {west}, North: {north}, South: {south}")
print(f"East: {neighbor_ranks['east']}, West: {neighbor_ranks['west']},",
      f"North: {neighbor_ranks['north']}, South: {neighbor_ranks['south']}\n")
