#!/bin/bash

MIN_SIZE=1000000
MAX_SIZE=128000000

MIN_RATIO=1
MAX_RATIO=4

echo "build size,probe size,key byte,selectivity,unique build key,joined size,build [ms],probe [ms],total [ms],num errors"

for (( b_size=${MIN_SIZE}; b_size<=${MAX_SIZE}; b_size=${b_size}*2 )); do
    for (( ratio=${MIN_RATIO}; ratio<=${MAX_RATIO}; ratio=${ratio}*2 )); do
        p_size=$((${ratio}*${b_size}))
        random=$((2*${b_size}))
        ./inner_join -build_tbl_size ${b_size} -build_tbl_loc D -probe_tbl_size ${p_size} -probe_tbl_loc D -joined_tbl_loc D -rand_max ${random} -s
    done
done