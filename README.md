# ABM
Agent based model framework to simulate collective foraging with visual private and social cues

## Introduction
This repository hold the code base for the agent based model framework implemented in python/pygame to model and simualate agents collectively foraging in the environment.

### Requirements
To run the simulations you will need python 3.8 or 3.9 and pip correspondingly. It is worth to set up a virtualenvironment using pipenv or venv for the project so that your global workspace is not polluted.

### Test
To test the code:
  1. Clone the repo
  2. Activate your virtual environment (pipenv, venv) if you are using one
  3. Move into the cloned repo where `setup.py` is located and run `pip install -e .` with that you installed the simulation package
  4. run the start entrypoint of the simulation package by running `abm-start`

## Install Grafana and InfluxDB
To monitor individual agents real time and save simulation data (i.e. write simulation data real time and save upon request at the end) we use InfluxDB and a grafana server for visualization. For this purpose you will need to install influx and grafana. If you don't do these steps you are still going to be able to run simulations, but you won't be able to save the resulting data or visualize the agent's parameters. This installation guide is only tested on Ubuntu. If you decide to use another op.system or you don't want to monitor and save simulation data, set `USE_IFDB_LOGGING` and `SAVE_CSV_FILES` parameters in the `.env` file to `0`.
<details>
  <summary>Click to expand for Grafana and InfluxDB installation details!</summary>
  
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
  
</details>

## Details of the package
In this section the package is detailed for reproducibility and for ease of use. Among others you can read about the main restrictions and assumptions we used in our framework, how one can initialize the package with different parameters through `.env` files, and how the code is structured.

### Code Elements
The code is structured into a single installable python package called `abm`. Submodules of this package contain the main classes and methods that are used to implement the functionalities of our framework, such as `Agent`, `Resource` and `Simulation` classes among others. A dedicated `contrib` submodule provides parameters for running the simulations in python syntax. These parameters are either initialized from a `.env` file (described later) or they are not to be changed throughout simulations (such as passwords and database details) and therefore fixed in these scripts. Note that although we store passwords as text these are absolutely insensitive as they are only needed locally on a simulation computer to access the database in which we store simulation data (that is by nature not sensitive data).

#### Submodules
The package includes the following submodules:

* `agent`: including the `Agent` class implementing a simple interactive agent that is able to move in the environment, update it's appearence in pygame, use visual social cues, find resource patches and exploit them. All necessary method that implements these behaviors are packed in the class and called in the `update` method that is used to update the agents status (position, orientation, exploited resources, decision parameters) in each timestep by pygame. This class inherits from the `pygame.Sprite` class and therefore can be used accordingly. A helper script of the submodule `supcalc.py` inlcudes some independent functions to calculate distances, norms, angles, etc.
* `contrib`: including helper parameters of the package that can be later imported as `abm.contrib.<name_of_param_bundle>`. For further information about what the individual parameter bundles include within this submodule please read the comment in the beginning of these scripts.
* `environment`: including classes and methods for environmental elements. As an example it includes the `Resource` class that implements a resource patch in the environment. Similar to the `Agent` class it inherits from pygame sprites and therefore the `update` method will call all relevant methods in each timestep.
* `loader`: including all classes and methods to dynamically load data that was generated with the package. These methods are for example cvs and json readers and initializers that initialize input classes for Replay and DataAnalysis tools.
* `metarunner`: including all classes and methods to run multiple simulations one after the other with programatically changed initialization parameters. The main classes are `Tunables` that define a criterion range with requested number of datapoints (e.g.: simulate with environment width from 300 to 600 via 4 datapoints). The `Constant` class that defines a fixed criterion throughout the simulations (e.g.: keep the simulation time `T` fixed at 1000). And `MetaProtocol` class that defines a batch of simulations with all combinations of the defined criteria according to the added `Tunable`s and `Constant`s. During running metaprotocols the corresponding initializations (as `.env` files) will be saved under the `data/metaprotocol/temp` folder. Only those `.env` files will be removed from here for which the simulations have been carried out, therefore the metaprotocol can be interrupted and finished later.
* `monitoring`: including all methods to interface with InfluxDB, Grafana and to save the stored data from the database at the end of the simulation. The data will be saved into the `data/simualtion_data/<timestamp_of_simulation>` of the root abm folder. The data will consist of 2 relevant csv files (agent_data, resource_data) containing time series of the agent and resource patch status and a json file containing all parameters of the simulation for reproducibility.
* `simulation`: including the main `Simulation` class that defines how the environment is visualized, what interactions the user can have with the pygame environment (e.g.: via cursor or buttons), and how the environment enforces some restrictions on agents, and how resources are regenerated. 

### Functionality and Behavior
Here you can read about how the framework works in large scale behavior and what restrictions and assumptions we used throughout the simulation.

#### Behavior
Upon starting the simulation an arena will pop up with given number of agents and resource patches. Details of these are controlled via `.env` variables and you can read more below. Agents will search for hidden resource patches and exploit/consume these with a given rate (resource unit/time) when found. Agents can behave according to 3 distinct behavioral states. These are Exploration (individual uninformed state looking for resources and integration individual and social cues in the meanwhile). Relocation (informed state in which the agent "decides" to join to another agent's patch. Exploitation (in which the agent consumes a resource patch and is recognized as a social visual cue for other agents). The mode of the agents are depicted with their colors, that is blue, purple and green respectively. Agents can collide with each other (red color) and in this case they avoid the collision by turning away from the other agents. Collision can be turned off during exploitation (ghost mode). Recognizing exploiting agents as social cues on the same patch can be turned off. Agents decide on which mode to enter via a dedicated decision process.
