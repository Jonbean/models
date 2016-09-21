#!/bin/bash

for i in `seq 10 29`; do
    qsub -pe serial 2 -l long adv$i;
done

