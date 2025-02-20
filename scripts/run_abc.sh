#!/bin/sh

#$ -N sim
#$ -cwd
#$ -j y
#$ -t 1-400
#$ -q short-centos79,long-centos79
#$ -l h_rt=20:00:00
#$ -l virtual_free=5G
#$ -o out_sim/$TASK_ID.out

singularity exec abc_container.sif python3 abc.py $SGE_TASK_ID
