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
influx --execute "grant all privileges on home to monitoring"
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
Upon starting the simulation an arena will pop up with given number of agents and resource patches. Details of these are controlled via `.env` variables and you can read more below. Agents will search for hidden resource patches and exploit/consume these with a given rate (resource unit/time) when found. The remaining resource units are shown on the patches as well as their quality (how much unit can be exploited in a time unit per agent). 

Agents can behave according to 3 distinct behavioral states. These are: Exploration (individual uninformed state looking for resources with random movement and integration of individual and social cues in the meanwhile). Relocation (informed state in which the agent "decides" to join to another agent's patch). Exploitation (in which the agent consumes a resource patch and is recognized as a social visual cue for other agents). The mode of the agents are depicted with their colors, that is blue, purple and green respectively. Agents can collide with each other (red color) and in this case they avoid the collision by turning away from the other agents. Collision can be turned off during exploitation (ghost mode). Recognizing exploiting agents as social cues on the same patch can be turned off. 

Each social cue (other exploiting agent) creates a visual projection on the focal agent's retina if in visual range (and the limited FOV allows). Relocation happens according to the overall excitation of the agent's retina. The focal agent steers right if the right hemifield is more excited and left if the left hemifield is more excited.

During exploitation agents slow down and stop on patches.

Agents decide on which mode to enter via a dedicated decision process. The decision process continously integrates private information (Did I find a new patch? How good the quality of the new patch is?) and social information (Do I see any other agents axploiting nearby? How many/how close according to visual projection field?). With the parameters of the decision process one con control how socially susceptible agents are and how much being in e.g. relocation inhibits exploitation and vica versa. Agents integrate infromation all the time and they can deliberately stop being in a behavioral mode to switch into another.

#### Interaction
During the simulation visualization can be turned off to speed up the run. In case it is turned on, the user is able to interact with the simulation as follows:

* click (left) and move agents in space
* rotate agents with mouse scroll
* pause/unpause simulation with `space`
* show social visual field with `return`
* increase/decrease framrate with `f`/`s`
* reset default framerate with `d`

##### Environment variables as parameters
To parametrize the simulation we use `.env` files. These include the main parameters line by line. This means, that a single `.env` file defines a simulation run fully. The env variables are as follows:

<details>
  <summary>Click to see all env variables!</summary>
  
* `N`: number of agents
* `N_RESOURCES`: number of resource patches
* `T`: number of simulation timesteps
* `INIT_FRAMERATE`: default framerate when visualization is on. Irrelevant for when visualization is turned off
* `WITH_VISUALIZATION`: turns visualization on or off
* `VISUAL_FIELD_RESOLUTION`: Resolution/size of agents' visual projection fields in pixels
* `ENV_WIDTH`: width of the environment in pixels
* `ENV_HEIGHT`: height of the environment in pixels
* `RADIUS_AGENT`: radius of agents in pixels
* `RADIUS_RESOURCE`: radius or resource patches in pixels
* `MIN_RESOURCE_PER_PATCH`: minimum contained resource units of a resourca patch. real value will be random uniform between min and max values.
* `MAX_RESOURCE_PER_PATCH`: maximum contained resource units of a resourca patch.
* `REGENERATE_PATCHES`: turns on or off resource patch regeneration upon full depletion.
* `AGENT_CONSUMPTION`: maximum resource consumption of agents (per time unit). Can be lower according to resource patch quality
* `MIN_RESOURCE_QUALITY`: minimum quality of resourca patch. real quality will be random uniform between min and max quality.
* `MAX_RESOURCE_QUALITY`: maximum quality of resource patches.
* `TELEPORT_TO_MIDDLE`: pulling exploiting agents into the middle of the resource patch if turned on.
* `GHOST_WHILE_EXPLOIT`: disabling collisions when the agents exploit when turned on.
* `PATCHWISE_SOCIAL_EXCLUSION`: not taking into consideration agents on the same patch as social cues if turned on.
* `AGENT_FOV`: Field of view of the agents. FOV is symmetric and defined with percent of pi. e.g if 0.6 then fov is (-0.6*pi, 0.6*pi). 1 is full 360 degree vision
* `VISION_RANGE`: visual range in pixels
* `VISUAL_EXCLUSION`: taking visual exclusion into account when calculating visual cues if turned on.
* `SHOW_VISUAL_FIELDS`: always show visual fields of agents when turned on.
* `SHOW_VISUAL_FIELDS_RETURN`: show visual fields of agents when return pressed if turned on
* `SHOW_VISION_RANGE`: visualizing visual range and field of view of agents when turned on.
* `USE_IFDB_LOGGING`: logs simulation data into a connected InfluxDB database when turned on (and InfluxDB is initialized)
* `SAVE_CSV_FILES`: saves data from connected InfluxDB instance as csv files if turned on.
  
 Parameters of the decision process as decsribed in rpopsal:
* `DEC_TW`: time constant of w process
* `DEC_EPSW`: social excitability
* `DEC_GW`: social decay
* `DEC_BW`: social process baseline
* `DEC_WMAX`: social process limit
* `DEC_TU`: time constant of u process
* `DEC_EPSU`: individual excitability
* `DEC_GU`: individual decay
* `DEC_BU`: individual process baseline
* `DEC_UMAX`: individual process limit
* `DEC_SWU`: social to individual inhibition
* `DEC_SUW`: individual to social inhibition
* `DEC_TAU`: novelty time window of private information
* `DEC_FN`: novelty multiplier
* `DEC_FR`: quality multiplier

Movement parameters:
* `MOV_EXP_VEL_MIN`: minimum exploration velocity
* `MOV_EXP_VEL_MAX`: maximum exploration velocity
* `MOV_EXP_TH_MIN`: minimum exploration orientation change (per time unit)
* `MOV_EXP_TH_MAX`: maximum exploration orientation change (per time unit)
* `MOV_REL_DES_VEL`: relocation velocity
* `MOV_REL_TH_MAX`: relocation maximal orientation change
* `CONS_STOP_RATIO`: deceleration during exploitation
  
</details>
