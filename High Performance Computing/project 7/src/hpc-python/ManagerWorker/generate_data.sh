#!/bin/bash

echo "tasks,workers,time" > scaling.csv

for s in 50 100; do 
    for r in 2 4 8 16; do
        time=$(mpiexec --oversubscribe -n $r python ./manager_worker.py 4001 4001 $s | grep "Run took" | awk '{print $3}')
        echo "$s,$r,$time" >> scaling.csv
    done
done
