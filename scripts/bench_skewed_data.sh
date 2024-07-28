#!/bin/bash

MIN_SKEW_MODE=1
MAX_SKEW_MODE=3

echo "build size,probe size,key byte,tuple byte,selectivity,unique build key,num part P1,num part P2,max part size factor,skewed mode,zipf factor,joined size,combined histo [ms],part P1 [ms],part P2 [ms],handle skew [ms],join [ms],total [ms],num errors"

for (( mode=${MIN_SKEW_MODE}; mode<=${MAX_SKEW_MODE}; mode=${mode}+1 )); do
    ./partitioned_join -b 32000000 -p 32000000 -r 64000000 -R -n 512 -N 512 -C -z 0.25 -Z ${mode}
    ./partitioned_join -b 32000000 -p 32000000 -r 64000000 -R -n 512 -N 512 -C -z 0.5 -Z ${mode}
    ./partitioned_join -b 32000000 -p 32000000 -r 64000000 -R -n 512 -N 512 -C -z 0.75 -Z ${mode}
    ./partitioned_join -b 32000000 -p 32000000 -r 64000000 -R -n 512 -N 512 -C -z 1 -Z ${mode}
done
