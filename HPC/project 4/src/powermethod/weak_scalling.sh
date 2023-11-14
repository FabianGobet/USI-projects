#!/bin/bash

echo "n,p,time" > weak_scalling.csv

p=(1 4 8 12 16 32)
n=(9600 19200 27152 33252 38400 54304)

for ((index=0; index<${#p[@]}; index++)); do
    p_i=${p[$index]}
    n_i=${n[$index]}
    mpirun --oversubscribe  -np $p_i ./powermethod $n_i >> weak_scalling.csv
done