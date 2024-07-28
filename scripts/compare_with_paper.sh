#!/bin/bash

../ICDE2019-GPU-Join/bin/release/bench -b 7 -a HJC -S 32000000 -R 32000000 
./partitioned_join -F -B ./unique_64000000.bin -P ./unique_64000000.bin -b 64000000 -p 64000000 -u -R -n 256 -N 256 -V -f 1 