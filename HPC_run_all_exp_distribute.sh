#!/bin/bash

# BEHAVIOR:
# Parses the abm/data/metarunner/experiments folder and runs each experiment py file in the folder on the HPC cluster.
# Each experiment will be run on a different node.
# Prepares the environment on the gateway such es env files, empty folders for logging and errors of the jobs, etc.

# Define how many repetitions should we simulate on the cluster in a heavily distributed manner,
# meaning we run only 1 batch per instance but we create many instances that can be distributed on the
# HPC cluster creating a parallel computation pool.
# To handle all the instances the original experiment file will be copied N times with random hash strings as
# well as N env files will be created. For each a new singularity instance will be dedicated. The saved data
# will be in the abm/data/simulation_data folder with the random hashes which then can be merged into a single experiment
# folder later.
NUM_INSTANCES_PER_EXP=10

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
  if [[ $exp_name == *"_"* ]]; then
    # if _ is given in the exp filename we skip it as it might be garbage from previous broken runs
    # so that we avoid exponential explosion of job numbers
    echo "Experiment is already hashed from previous runs, skipping it..."
  else
    exp_path_array+=("$exp_path")
    exp_name_array+=("$exp_name")
  fi
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
    echo "Copy experiment with hash"
    cp $search_dir/$exp_name.py $search_dir/$exp_name_hashed.py

    # Run an experiment on a dedicated node
    echo "Starting experiment $exp_name_hashed"
    sbatch --export=EXPERIMENT_NAME=$exp_name_hashed,HPC_DISTRIBUTED_ABM=yes ./HPC_batch_run.sh

  done
done
