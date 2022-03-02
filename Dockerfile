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
RUN apt-get install -y xvfb libglib2.0-0 libsm6 libxrender1 libxext6 sudo

# Install package base requirements
RUN pip install virtualenv pip

# Copy package to docker and install it
COPY . /app
WORKDIR /app
# Create entrypoint
RUN chmod +x ./docker_entrypoint.sh

# Install p34ABM
RUN pip install -e .

# Add a non-root user so that the generated data can be easily handled on host
ENV GID 1000
ENV UID 1002
RUN groupadd --gid $GID appgroup && \
    useradd -r -d /app -g appgroup -G root,sudo -u $UID appuser

RUN adduser appuser sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Change the owner of the application folder to new user
RUN chown -R appuser:appgroup /app

# Change user to new user (from this on, we need to use sudo for root methods)
USER appuser
CMD ["./docker_entrypoint.sh"]