#!/bin/bash

#SBATCH --job-name=scioi_p34_ABM_simulation
#SBATCH --output=slurm_log/._log_%j.log         # output file
#SBATCH --error=slurm_log/._err_%j.err          # error file
#SBATCH --partition=ex_scioi_node   # partition to submit to
#SBATCH --ntasks=1
#SBATCH --time=1-00:00          # Runtime in D-HH:MM
#SBATCH --cpus-per-task=2
#SBATCH --exclusive

# BEHAVIOR:
# Runs a single experiment on a dedicated cluster node. The experiment is defined in the dedicated folder of the codebase with the
# metarunner API of the ABM framework.

# PREPARATION: (Only needed if this script is directly called with sbatch. Otherwise run HPC_run_all_exp.sh.)
# - We suppose that the git repo is already cloned and we are inside the cloned folder root where this file is located
# - If git does not work and the repo is not yet cloned use wget with the git codebase archive
#   and the develop branch name, then unzip the folder
# - We suppose that a singularity image has been built from the DockerHub image on a Linux host system where sudo
#   privileges are provided
# - We suppose that the resulting SIF file has a filename of scioip34abm.sif and is copied into the codebase repo
# - We call this script with sbatch on all dedicated nodes passing the environment variable EXPERIMENT_NAME
# - we assume that there is an EXPERIMENT_NAME.py file in abm/data/metarunner/experiments folder and a
#   corresponding EXPERIMENT_NAME.env file in the root codebase folder.
# - we assume there is a slurm_log folder where logs and errors are forwarded

# Showing experiment name
echo "Starting experiment on node with experiment name: $EXPERIMENT_NAME"
echo "Current path: $(pwd)"

# Checking for singularity container
if [ -f "scioip34abm.sif" ]; then
  echo "Found candidate SIF file for singularity runs as $(pwd)\scioip34abm.sif"
else
  echo "Didn't find scioip34abm.sif in the codebase folder. Please be sure that the SIF filename matches and is copied into the gateway path: $(pwd)"
  exit 1
fi

# Initializing empty bindmounts if not yet done
if [ ! -d "influxdb" ]; then
  mkdir influxdb
  echo "Bind root directory created for influxdb..."
fi

if [ ! -d "influxdb/influxdb_$EXPERIMENT_NAME" ]; then
  mkdir influxdb/influxdb_$EXPERIMENT_NAME
  echo "Bind directory created for influxdb $EXPERIMENT_NAME..."
fi

if [ ! -d "journal" ]; then
  mkdir journal
  echo "Bind directory created for influxdb logging..."
fi

if [ ! -d "journal/journal_$EXPERIMENT_NAME" ]; then
  mkdir journal/journal_$EXPERIMENT_NAME
  echo "Bind directory created for influxdb $EXPERIMENT_NAME..."
fi


# Load singularity
echo "Loading singularity..."
module load singularity/3.7.0
singularity version

# Create singularity instance
echo "Creating singularity instance..."
singularity instance start --bind "/$(pwd):/app" \
                           --bind "/$(pwd)/influxdb/influxdb_$EXPERIMENT_NAME:/var/lib/influxdb" \
                           --bind "/$(pwd)/journal/journal_$EXPERIMENT_NAME:/var/log/journal" \
                           scioip34abm.sif scioip34abmcontainer_$EXPERIMENT_NAME

# Executing experiment via entrypoint on node inside singularity instance
echo "Starting experiment on instance..."
env SINGULARITYENV_EXPERIMENT_NAME=$EXPERIMENT_NAME singularity exec instance://scioip34abmcontainer_$EXPERIMENT_NAME sh /app/singularity_entrypoint.sh

# When done we clean up
echo "Experiment finished, stopping singularity instance."
singularity instance stop scioip34abmcontainer_$EXPERIMENT_NAME

# We exit with success
exit