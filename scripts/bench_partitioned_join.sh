#!/bin/bash

MIN_SIZE=4000000
MAX_SIZE=128000000

MIN_RATIO=1
MAX_RATIO=1

MIN_PART=256
MAX_PART=256

echo "build size,probe size,key byte,tuple byte,selectivity,unique build key,num part P1,num part P2,max part size factor,skewed mode,zipf factor,joined size,combined histo [ms],part P1 [ms],part P2 [ms],handle skew [ms],join [ms],total [ms],num errors"

for (( b_size=${MIN_SIZE}; b_size<=${MAX_SIZE}; b_size=${b_size}*2 )); do
    for (( ratio=${MIN_RATIO}; ratio<=${MAX_RATIO}; ratio=${ratio}*2 )); do
        p_size=$((${ratio}*${b_size}))
        random=$((2*${b_size}))
        for (( part=${MIN_PART}; part<=${MAX_PART}; part=${part}*2 )); do 
            ./partitioned_join -b ${b_size} -p ${p_size} -r ${random} -R -n ${part} -N ${part} -u -C
        done 
    done
done

for (( b_size=${MIN_SIZE}; b_size<=${MAX_SIZE}; b_size=${b_size}*2 )); do
    for (( ratio=${MIN_RATIO}; ratio<=${MAX_RATIO}; ratio=${ratio}*2 )); do
        p_size=$((${ratio}*${b_size}))
        random=$((2*${b_size}))
        for (( part=${MIN_PART}; part<=${MAX_PART}; part=${part}*2 )); do 
            ./partitioned_join -b ${b_size} -p ${p_size} -r ${random} -R -n ${part} -N ${part} -u -C -V
        done 
    done
done
