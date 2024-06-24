#!/bin/bash

for j in {9..10}; do
    i=10
    while [ $i -le 100000 ]; do
        ./recur_omp $i $j
        i=$((i * 10))
    done
done

