#!/bin/bash

# We assume that the singularity instance has been created in a way that both the data saving folder
# and the necessary influxdb folders are binded as volumes with the host so the container can be written.
# e.g.: singularity instance start --bind "/$(pwd):/app"
#                                  --bind "/$(pwd)/influxdb:/var/lib/influxdb" \
#                                  --bind "/$(pwd)/journal:/var/log/journal" scioip34abm.sif scioip34abmcontainer
# 1.) the first bind will bind the application folder so that we can replace and add new files, environment files, etc.
# 2.) the second bind will bind the influxdb data storage (that might be shared across instances
# 3.) the third bind will bind influx logs so that the service can be started without sudo

# Then in the singularity instance we need to start the influxdb service with influxd (output fowarded so we get back
# the terminal access) then initialize the database
influxd > /dev/null 2>&1 & disown && \
sleep 2 && \
influx --execute "create database home" && \
influx --execute "use home" && \
influx --execute "create user monitoring with password 'password' with all privileges" && \
influx --execute "grant all privileges on home to monitoring" && \
influx --execute "show users"

# The experiment name can be passed for each instance individually so that different env files will be used but
# in the same attached volume from all nodes
EXPERIMENT_NAME=$EXPERIMENT_NAME python3 -u /app/abm/data/metaprotocol/experiments/docker_exp.py

# So to start an experiment in a given singularity instance use
# e.g.: env SINGULARITYENV_EXPERIMENT_NAME=docker_exp singularity exec instance://<instance name> sh /app/singularity_entrypoint.sh