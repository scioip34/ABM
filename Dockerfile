FROM ubuntu:20.04

COPY . /app
WORKDIR /app

RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update && apt-get install -y python3-pip
RUN pip install virtualenv pip
RUN ls -a
RUN pip install .


CMD python -V