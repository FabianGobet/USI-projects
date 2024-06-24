#!/bin/bash

echo "n,p,time" > strong_scalling.csv

for j in 1 4 8 12 16 32; do 
    mpirun --oversubscribe  -np $j ./powermethod >> strong_scalling.csv
done