#!/bin/bash

sudo systemctl unmask influxdb.service && \
sudo systemctl start influxdb && \
sudo systemctl enable influxdb.service

EXPERIMENT_NAME=docker_exp python3 -u /app/abm/data/metaprotocol/experiments/docker_exp.py