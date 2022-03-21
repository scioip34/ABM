#!/bin/bash

# BEHAVIOR:
# Parses the abm/data/metarunner/experiments folder and runs each experiment py file in the folder on the HPC cluster.
# Each experiment will be run on a different node.
# Prepares the environment on the gateway such es env files, empty folders for logging and errors of the jobs, etc.

NUM_INSTANCES_PER_EXP=3

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
  echo "Handling instances for base experiemnt $exp_name, creating $NUM_INSTANCES_PER_EXP instances"
  for i in $(seq 1 $NUM_INSTANCES_PER_EXP)
  do
    # Generating random hash for each instance per experiment
    random_hash=$(echo $RANDOM | md5sum | head -c 20)

    # Noting data with experiment name and hash
    exp_name_hashed=$exp_name"_"$random_hash
    if [ ! -f "./$exp_name_hashed.env" ]; then
      cp ./.env ./$exp_name_hashed.env
      echo "Created default env file for experiment $exp_name_hashed"
    else
      echo "Env file already exists for experiment $exp_name_hashed"
    fi

    # Preparing temporary exp file with hash
    cp $search_dir/$exp_name.py $search_dir/$exp_name_hashed.py

    # Run an experiment on a dedicated node
    echo "Starting experiment $exp_name_hashed"
    sbatch --export=EXPERIMENT_NAME=$exp_name_hashed ./HPC_batch_run.sh

    # Cleaning up
    # remove env file
    rm ./$exp_name_hashed.env
    # remove temp exp file
    rm $search_dir/$exp_name_hashed.py
  done
done
