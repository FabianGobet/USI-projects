#!/bin/bash

echo "threads,ranks,grid,time" > scaling.csv

for t in {1..10}; do 
    export OMP_NUM_THREADS=$t
    for r in 1 2 4; do
        for g in 128 256 512 1024; do
            time=$(mpirun --oversubscribe -np $r ./main $g $g 100 0.01 | grep "simulation took" | awk '{print $3}')
            echo "$t,$r,$g,$time" >> scaling.csv
        done
    done
done
