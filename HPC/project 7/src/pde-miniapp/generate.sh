#!/bin/bash

export TMPDIR=/tmp

# Strong scaling for fixed grid sizes
echo "ranks,grid_size,iters_per_second" > strong_scaling.csv
for grid_size in 128 256 512 1024; do
    for ranks in 1 2 4 8 16 32; do
        time=$(mpirun --oversubscribe -np $ranks ./main $grid_size 100 0.005 | grep "iters/second" | awk "{print $8}")
        echo "$grid_size,$ranks,$time" >> strong_scaling.csv
    done
done

calculate_processes() {
    local size=$1
    echo $(( (size / 128) ** 2 ))
}

grid_sizes=(128 256 512 1024)
echo "ranks,grid_size,iters_per_second" > weak_scaling.csv

for size in "${grid_sizes[@]}"; do
    ranks=$(calculate_processes $size)
    time=$(mpirun --oversubscribe -np $ranks ./main $size 100 0.005 | grep "iters/second" | awk '{print $8}')
    echo "$size,$ranks,$time" >> weak_scaling.csv
done

