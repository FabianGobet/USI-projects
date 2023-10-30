#!/bin/bash

for j in 64 128 256 512 1024; do
  for i in {1..24}; do
    export OMP_NUM_THREADS=$i
    ./main $j 100 0.005
  done
done