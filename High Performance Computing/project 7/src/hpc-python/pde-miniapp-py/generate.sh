#!/bin/bash

export TMPDIR=/tmp
export DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/open-mpi/4.1.6/lib:$DYLD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH


# Strong scaling for fixed grid sizes
echo "ranks,grid_size,iters_per_second" > strong_scaling_python.csv
for grid_size in 128 256 512 1024; do
    for ranks in 1 2 4 8 16 32; do
        time=$(mpiexec --oversubscribe -n $ranks python ./main.py $grid_size 100 0.005 | grep "iters/second" | awk '{print $8}')
        echo "$ranks,$grid_size,$time" >> strong_scaling_python.csv
    done
done


calculate_processes() {
    local size=$1
    echo $(( (size / 128) ** 2 ))
}

grid_sizes=(128 256 512 1024)
echo "ranks,grid_size,iters_per_second" > weak_scaling_python.csv

for size in "${grid_sizes[@]}"; do
    ranks=$(calculate_processes $size)
    time=$(mpiexec --oversubscribe -n $ranks python ./main.py $size 100 0.005 | grep "iters/second" | awk '{print $8}')
    echo "$size,$ranks,$time" >> weak_scaling_python.csv
done

