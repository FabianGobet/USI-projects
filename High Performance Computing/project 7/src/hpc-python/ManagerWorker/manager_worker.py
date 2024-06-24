from mandelbrot_task import *
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpi4py import MPI # MPI_Init and MPI_Finalize automatically called
import numpy as np
import sys
import time

# some parameters
MANAGER = 0       # rank of manager
TAG_TASK      = 1 # task       message tag
TAG_TASK_DONE = 2 # tasks done message tag
TAG_DONE      = 3 # done       message tag

def manager(comm, tasks):
    """
    The manager.

    Parameters
    ----------
    comm : mpi4py.MPI communicator
        MPI communicator
    tasks : list of objects with a do_task() method perfroming the task
        List of tasks to accomplish

    Returns
    -------
    TasksDoneByWorker : 
        list of data from messages received from workers
    """
    size = comm.Get_size()
    num_tasks = len(tasks)

    send_index = 0
    for i in range(1, min(size,num_tasks)):
        comm.send(tasks[send_index], dest=i, tag=TAG_TASK)
        send_index += 1

    receive_index = 0
    TasksDoneByWorker = [[] for _ in range(size)]
    while send_index < num_tasks:
        status = MPI.Status()
        data = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_TASK_DONE, status=status)
        TasksDoneByWorker[status.Get_source()].append(data)
        comm.send(tasks[send_index], dest=status.Get_source(), tag=TAG_TASK)
        send_index += 1
        receive_index +=1

    while(receive_index<send_index):
        status = MPI.Status()
        data = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_TASK_DONE, status=status)
        TasksDoneByWorker[status.Get_source()].append(data)
        receive_index +=1

    for i in range(1, size):
        comm.send(None, dest=i, tag=TAG_DONE)

    return TasksDoneByWorker

def worker(comm):

    """
    The worker.

    Parameters
    ----------
    comm : mpi4py.MPI communicator
        MPI communicator
    """
    while True:
        status = MPI.Status()
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == TAG_DONE:
            break
        task.do_work()
        comm.send(task, dest=0, tag=TAG_TASK_DONE)

def readcmdline(rank):
    """
    Read command line arguments

    Parameters
    ----------
    rank : int
        Rank of calling MPI process

    Returns
    -------
    nx : int
        number of gridpoints in x-direction
    ny : int
        number of gridpoints in y-direction
    ntasks : int
        number of tasks
    """
    # report usage
    if len(sys.argv) != 4:
        if rank == MANAGER:
            print("Usage: manager_worker nx ny ntasks")
            print("  nx     number of gridpoints in x-direction")
            print("  ny     number of gridpoints in y-direction")
            print("  ntasks number of tasks")
        sys.exit()

    # read nx, ny, ntasks
    nx = int(sys.argv[1])
    if nx < 1:
        sys.exit("nx must be a positive integer")
    ny = int(sys.argv[2])
    if ny < 1:
        sys.exit("ny must be a positive integer")
    ntasks = int(sys.argv[3])
    if ntasks < 1:
        sys.exit("ntasks must be a positive integer")

    return nx, ny, ntasks

if __name__ == "__main__":

    # get COMMON WORLD communicator, size & rank
    comm    = MPI.COMM_WORLD
    size    = comm.Get_size()
    my_rank = comm.Get_rank()

    # report on MPI environment
    if my_rank == MANAGER:
        print(f"MPI initialized with {size:5d} processes")

    # read command line arguments
    nx, ny, ntasks = readcmdline(my_rank)

    
    if my_rank == MANAGER:
        x_min = -2.
        x_max  = +1.
        y_min  = -1.5
        y_max  = +1.5
        timespent = - time.perf_counter()

        M = mandelbrot(x_min, x_max, nx, y_min, y_max, ny, ntasks)
        tasks = M.get_tasks()

        TasksDoneByWorker = manager(comm, tasks)
        m = M.combine_tasks([item for sublist in TasksDoneByWorker for item in sublist])

        plt.imshow(m.T, cmap="gray", extent=[x_min, x_max, y_min, y_max])
        plt.savefig("mandelbrot.png")

        timespent += time.perf_counter()
        print(f"Run took {timespent:f} seconds")
        for i in range(size):
            if i == MANAGER:
                continue
            print(f"Process {i:5d} has done {len(TasksDoneByWorker[i]):10d} tasks")
        print("Done.")

    else:
        worker(comm)
