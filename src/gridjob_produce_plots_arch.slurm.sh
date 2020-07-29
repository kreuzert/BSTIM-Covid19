#!/bin/bash

# Arch script for generating day-specific scripts.
# The DATE_ID variable will be updated to the correct index in this case.
DATE_ID=56

# create task output directory
TASK_ID=$((${SLURM_ARRAY_TASK_ID}+${SLURM_LOCALID}))
TASK_DIR=${SCRATCH}/run_${SLURM_ARRAY_JOB_ID}/task_${TASK_ID}
mkdir -p ${TASK_DIR}
echo "TASK ${TASK_ID}: Running in job-array ${SLURM_ARRAY_JOB_ID} on `hostname` and dump output to ${TASK_DIR}"

# activate virtual python environment
source ${PROJECT}/.local/share/venvs/covid19dynstat_v01/bin/activate

export DATE_ID=${DATE_ID}
THEANO_FLAGS="base_compiledir=${TASK_DIR}/,floatX=float32,device=cpu,openmp=True,mode=FAST_RUN,warn_float64=warn" python3 produce_plots.py > ${TASK_DIR}/log.txt
