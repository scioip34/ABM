#!/bin/bash

# BEHAVIOR:
# Parses the abm/data/metarunner/experiments folder and runs each experiment py file in the folder on the HPC cluster.
# Each experiment will be run on a different node.
# Prepares the environment on the gateway such es env files, empty folders for logging and errors of the jobs, etc.

# Initializing SLURM logging structure
if [ ! -d "slurm_log" ]; then
  mkdir slurm_log
  echo "Directory for job logs initialized..."
fi

# Parsing experiment files
search_dir=./abm/data/metaprotocol/experiments
if [ ! -d "$search_dir" ]; then
  echo "Can not find experiments folder..."
  exit 1
else
  echo "Found experiment folder $search_dir"
fi

# Find experiment files and their names
exp_path_array=()
exp_name_array=()
for exp_path in "$search_dir"/*.py
do
  exp_name=$(basename $exp_path .py)
  echo "Found experiment: $exp_name"
  exp_path_array+=("$exp_path")
  exp_name_array+=("$exp_name")
done

# Prepare empty env files for each experiment on root
for exp_name in "${exp_name_array[@]}"
do
  if [ ! -f "./$exp_name.env" ]; then
    cp ./.env ./$exp_name.env
    echo "Created default env file for experiment $exp_name"
  else
    echo "Env file already exists for experiment $exp_name"
  fi
done

# Run an experiment on a dedicated node
for exp_name in "${exp_name_array[@]}"
do
  echo "Starting experiment $exp_name"
  sbatch --export=EXPERIMENT_NAME=$exp_name ./HPC_batch_run.sh
  echo "Wait a few secs..."
  sleep 5
done