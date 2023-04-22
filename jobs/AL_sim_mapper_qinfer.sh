#!/bin/bash

# Initialize the module command first
source /etc/profile

# Load Conda environment
conda activate active-qinfer-hl-env

echo "My run number: " $1

# Call your script as you would from the command line passing $1 and $2 as arguments
python AL_FIR_runner_ibmq_sim_long_qinfer.py --run_seed=$1
