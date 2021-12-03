# ABM
Agent based model framework to simulate collective foraging with visual private and social cues

## Introduction
This repository hold the code base for the agent based model framework implemented in python/pygame to model and simualate agents collectively foraging in the environment.

## Requirements
To run the simulations you will need python 3.8 or 3.9 and pip correspondingly. It is worth to set up a virtualenvironment using pipenv or venv for the project so that your global workspace is not polluted.

## Test
To test the code:
  1. Clone the repo
  2. Activate your virtual environment (pipenv, venv) if you are using one
  3. Move into the cloned repo where `setup.py` is located and run `pip install -e .` with that you installed the simulation package
  4. run the start entrypoint of the simulation package by running `abm-start`

## Install Grafana and InfluxDB
To monitor individual agents real time and save simulation data (i.e. write simulation data real time and save upon request at the end) we use InfluxDB and a grafana server for visualization. For this purpose you will need to install influx and grafana. If you don't do these steps you are still going to be able to run simulations, but you won't be able to save the resulting data or visualize the agent's parameters. This installation guide is only tested on Ubuntu. If you decide to use another op.system or you don't want to monitor and save simulation data, set `USE_IFDB_LOGGING` and `SAVE_CSV_FILES` parameters in the `.env` file to `0`.

### Install Grafana
1. run the following commands to add the grafana APT repository and install grafana
```bash
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
echo "deb https://packages.grafana.com/oss/deb stable main" | sudo tee -a /etc/apt/sources.list.d/grafana.list
sudo apt-get update
sudo apt-get install -y grafana
```
2. enable and start the grafana server
```bash
sudo /bin/systemctl enable grafana-server
sudo /bin/systemctl start grafana-server
```

3. as we will use real time monitoring we have to change the minimal graph refresh rate in the config file of grafana.
   1. use `sudo nano /etc/grafana/grafana.ini` to edit the config file
   2. use `Ctrl` + `W` to serach for the term `min_refresh_interval`
   3. change the value from `5s` to `100ms`
   4. delete the commenting `;` character from the beginning of the row
   5. save the file

4. restart your computer with `sudo reboot`
5. you can now check your installation. Open a browser on the client PC and go to `http://localhost:3000`. You’re greeted with the Grafana login page.
6. Log in to Grafana with the default username `admin`, and the default `password` admin.
7. Change the password for the admin user when asked.

### Install influxdb:
1. Use the following commands to add InfluxDB APT repository and install InfluxDB
```bash
wget -qO- https://repos.influxdata.com/influxdb.key | sudo apt-key add -
source /etc/os-release
echo "deb https://repos.influxdata.com/debian $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/influxdb.list
sudo apt update && sudo apt install -y influxdb
```

2. Start and enable the service
```bash
sudo systemctl unmask influxdb.service
sudo systemctl start influxdb
sudo systemctl enable influxdb.service
```
3. Use the following commands to initialize a home InfluxDB instance and grant priviliges to grafana. Please note that in general passwords should not be uploaded to github. We are doing it now as this process is not sensitive (saving simulation data on local database) and doesn't make sense to parametrize the password.
```bash
influx --execute "create database home"
influx --execute "use home"
influx --execute "create user monitoring with password 'password' with all privileges"
influx --execute "grant all privileges on home to grafana"
influx --execute "show users"
```
4. after the last command you will see this
> user admin
> ---- -----
> grafana true

### Connect Grafana with Influx
(the following instructions were copied from Step4. of [this source](https://simonhearne.com/2020/pi-influx-grafana/#step-4-add-influx-as-a-grafana-data-source))

> Now we have both Influx and Grafana running, we can stitch them together. Log in to your Grafana instance and head to “Data Sources”. Select “Add new Data Source” and find InfluxDB under “Timeseries Databases”.

> As we are running both services on the same Pi, set the URL to localhost and use the default influx port of 8086:
> 
> [Image](https://simonhearne.com/images/grafana1.png)
> 
> We then need to add the database (home), user (monitoring) and password (password) that we set earlier:
> 
> [Image](https://simonhearne.com/images/grafana2.png)
> 
> That’s all we need! Now go ahead and hit “Save & Test” to connect everything together. You will see a "Data source is working" message

### Import Dashboard from repo
1. Open your grafana app from the browser and on the left menu bar click on the "+" button and the on the "Import button"
2. Upload the json file (that holds the blueprint of the grafana dashboard) from the repo under the path `abm/data/grafana_dashboard.json`
