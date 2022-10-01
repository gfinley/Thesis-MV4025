#!/bin/bash
n="2"
echo "Spawning $n processes"
for ((i=1; i <= $n; i++))
do
    # sleep needed to avoid redundant runs because named mutexes not working on Linux
    ( sleep 0.25 )
    ( bash run_process.sh $1 ) &
done
