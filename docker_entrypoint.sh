#!/bin/bash

sudo systemctl unmask influxdb.service && \
sudo systemctl start influxdb && \
sudo systemctl enable influxdb.service

# Replace the app.py in the root folder with the app.py in the project folder
# if the environment variable PROJECT is not empty
if [ ! -z "$PROJECT" ]; then
  echo "Replacing app.py with app.py from project $PROJECT"
  rm -f ./app/abm/app.py && \
  cp /abm/projects/$PROJECT/app.py /app/abm/app.py
fi

EXPERIMENT_NAME=docker_exp python3 -u /app/abm/data/metaprotocol/experiments/docker_exp.py