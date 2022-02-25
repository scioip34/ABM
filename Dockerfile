FROM ubuntu:20.04

# Install base requirements
RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update && \
    apt-get install -y python3-pip wget apt-utils systemctl

# Download and install InlfuxDB
RUN wget -qO- https://repos.influxdata.com/influxdb.key | apt-key add - && \
    echo "deb https://repos.influxdata.com/debian bionic stable" | tee /etc/apt/sources.list.d/influxdb.list && \
    apt update && apt install -y influxdb

# Setup database for data storage
RUN systemctl unmask influxdb.service && \
    systemctl start influxdb && \
    systemctl enable influxdb.service && \
    influx --execute "create database home" && \
    influx --execute "use home" && \
    influx --execute "create user monitoring with password 'password' with all privileges" && \
    influx --execute "grant all privileges on home to monitoring" && \
    influx --execute "show users"

# Preparing headless mode requirements
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin
RUN apt-get install -y xvfb libglib2.0-0 libsm6 libxrender1 libxext6

# Install package base requirements
RUN pip install virtualenv pip

# Copy package to docker and install it
COPY . /app
WORKDIR /app
RUN ls -a

# Install p34ABM
RUN pip install -e .

# Create entrypoint
CMD ["headless-abm-start"]