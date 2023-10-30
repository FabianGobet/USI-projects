#!/bin/bash


for i in 8 16 32 64; do
    for j in {1..24}; do
        if (( i * i % j == 0 )); then
            t_i=$i
            t_j=$j
            while (( t_i <= 1024 && t_j <= 24 )); do
                export OMP_NUM_THREADS=$t_j
                echo -n "$i,$j," >> mydata.csv
                ./main $t_i 100 0.005
                t_i=$((t_i * 2))
                t_j=$((t_j * 4))
            done
        fi
    done
done