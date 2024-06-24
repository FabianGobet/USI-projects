#!/bin/bash

for t in 7; do 
    export OMP_NUM_THREADS=$t
    for r in 2; do
        for g in 128; do
            mpirun --oversubscribe -np $r ./main $g $g 100 0.01
        done
    done
done