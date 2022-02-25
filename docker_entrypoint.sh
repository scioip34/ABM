#!/bin/bash

systemctl unmask influxdb.service && \
systemctl start influxdb && \
systemctl enable influxdb.service

headless-abm-start