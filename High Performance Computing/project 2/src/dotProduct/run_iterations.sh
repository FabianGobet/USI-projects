#!/bin/bash

for i in {0..3}; do
    n=$((100000 * (10 ** i)))
    for j in {1..12}; do
        ./dotProduct $j $n
    done
done
