#!/bin/bash

#$ -S /bin/bash
#$ -M jontsai@uchicago.edu
#$ -N HierBLSTMAtt
#$ -m beasn
#$ -r n
#$ -o HierBLSTMAtt.out
#$ -e HierBLSTMAtt.err
#$ -cwd 

SETTING1="300" 
SETTING2="0.0"
SETTING3="20" 
SETTING4="0.25"

export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
echo "Start - `date`"
/home-nfs/jontsai/anaconda/bin/python BLSTM_attention_dev.py \
$SETTING1 $SETTING2 $SETTING3 $SETTING4
echo "Finish - `date`"

