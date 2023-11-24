#!/bin/bash


for t in 5; do 
    export OMP_NUM_THREADS=$t
    for r in 3 4; do
        for g in 128 256 512 1024; do
            time=$(mpirun --oversubscribe -np $r ./main $g $g 100 0.01 | grep "simulation took" | awk '{print $3}')
            echo "$t,$r,$g,$time" >> scaling.csv
        done
    done
done
