"""
@author: mezdahun
@description: Helper functions for InfluxDB
"""
import datetime
import os
import numpy as np

from influxdb import InfluxDBClient, DataFrameClient

import abm.contrib.ifdb_params as ifdbp


def create_ifclient():
    """Connecting to the InfluxDB defined with environmental variables and returning a client instance.
        Args:
            None
        Returns:
            ifclient: InfluxDBClient connected to the database defined in the environment variables."""
    ifclient = InfluxDBClient(ifdbp.INFLUX_HOST,
                              ifdbp.INFLUX_PORT,
                              ifdbp.INFLUX_USER,
                              ifdbp.INFLUX_PSWD,
                              ifdbp.INFLUX_DB_NAME)
    return ifclient


def pad_to_n_digits(number, n=3):
    """
    Padding a single number to n digits with leading zeros so that lexicographic sorting does not mix fields of a
    measurement in InfluxDb.
        Args:
            number: int or string of a number
            n: the number of desired digits of the output
        Returns:
            padded number or the input number if it already has the desired length
    """
    len_diff = n - len(str(number))
    if len_diff > 0:
        return len_diff * '0' + str(number)
    else:
        return str(number)


def save_agent_data(ifclient, agents):
    """Saving relevant agent data into InfluxDB intance"""
    measurement_name = "agent_data"
    fields = {}
    for agent in agents:
        agent_name = f"agent-{pad_to_n_digits(agent.id, n=2)}"
        # take a timestamp for this measurement
        time = datetime.datetime.utcnow()

        # format the data as a single measurement for influx
        fields[f"posx_{agent_name}"] = int(agent.position[0])
        fields[f"posy_{agent_name}"] = int(agent.position[1])
        fields[f"orientation_{agent_name}"] = float(agent.orientation)
        fields[f"velocity_{agent_name}"] = float(agent.velocity)
        fields[f"w_{agent_name}"] = float(agent.w)
        fields[f"u_{agent_name}"] = float(agent.u)
        fields[f"Ipriv_{agent_name}"] = float(agent.I_priv)
        fields[f"mode_{agent_name}"] = int(mode_to_int(agent.mode))
        fields[f"collectedr_{agent_name}"] = float(agent.collected_r)
        fields[f"expl_patch_id_{agent_name}"] = int(agent.exploited_patch_id)
        # only storing visual field edges to compress data and keep real time simulations
        fields[f"vfield_up_{agent_name}"] = f"{np.where(np.roll(agent.soc_v_field,1) < agent.soc_v_field)[0]}"
        fields[f"vfield_down_{agent_name}"] = f"{np.where(np.roll(agent.soc_v_field, 1) > agent.soc_v_field)[0]}"

    body = [
        {
            "measurement": measurement_name,
            "time": time,
            "fields": fields
        }
    ]

    # write the measurement
    ifclient.write_points(body)


def mode_to_int(mode):
    """converts a string agent mode flag into an int so that it can be saved into a fluxDB"""
    if mode == "explore":
        return int(0)
    elif mode == "exploit":
        return int(1)
    elif mode == "relocate":
        return int(2)
    elif mode == "collide":
        return int(3)


def save_resource_data(ifclient, resources):
    """Saving relevant resource patch data into InfluxDB instance"""
    measurement_name = "resource_data"
    fields = {}
    for res in resources:
        res_name = f"res-{pad_to_n_digits(res.id, n=3)}"
        # take a timestamp for this measurement
        time = datetime.datetime.utcnow()
        # format the data as a single measurement for influx
        # pos and radius are enough to calculate center
        # (only important in spatially moving res, otherwise take the first element)
        # wasteful with resources but generalizable for later with no effort
        fields[f"posx_{res_name}"] = int(res.position[0])
        fields[f"posy_{res_name}"] = int(res.position[1])
        fields[f"radius_{res_name}"] = int(res.radius)

        fields[f"resc_left_{res_name}"] = float(res.resc_left)
        fields[f"quality_{res_name}"] = float(res.unit_per_timestep)

    body = [
        {
            "measurement": measurement_name,
            "time": time,
            "fields": fields
        }
    ]

    # write the measurement
    ifclient.write_points(body)


def save_simulation_params(ifclient, sim):
    """saving simulation parameters to IFDB"""

    measurement_name = "simulation_params"
    fields = {}

    # take a timestamp for this measurement
    time = datetime.datetime.utcnow()

    # format the data as a single measurement for influx
    fields["vision_range"] = sim.vision_range
    fields["width"] = sim.WIDTH
    fields["height"] = sim.HEIGHT
    fields["window_pad"] = sim.window_pad
    fields["N_agents"] = sim.N
    fields["T"] = sim.T
    fields["agent_radii"] = sim.agent_radii
    fields["v_field_res"] = sim.v_field_res
    fields["pooling_time"] = sim.pooling_time
    fields["pooling_prob"] = sim.pooling_prob
    fields["agent_consumption"] = sim.agent_consumption
    fields["teleport_exploit"] = sim.teleport_exploit
    fields["visual_exclusion"] = sim.visual_exclusion
    fields["N_resc"] = sim.N_resc
    fields["resc_radius"] = sim.resc_radius
    fields["min_resc_units"] = sim.min_resc_units
    fields["max_resc_units"] = sim.max_resc_units
    fields["regenerate_resources"] = sim.regenerate_resources

    body = [
        {
            "measurement": measurement_name,
            "time": time,
            "fields": fields
        }
    ]

    # write the measurement
    ifclient.write_points(body)


def save_ifdb_as_csv():
    """Saving the whole influx database as a single csv file"""
    # from influxdb_client import InfluxDBClient
    ifclient = DataFrameClient(ifdbp.INFLUX_HOST,
                              ifdbp.INFLUX_PORT,
                              ifdbp.INFLUX_USER,
                              ifdbp.INFLUX_PSWD,
                              ifdbp.INFLUX_DB_NAME)

    # create base folder in data
    root_abm_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    save_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = os.path.join(root_abm_dir, ifdbp.SAVE_DIR, save_timestamp)
    os.makedirs(save_dir, exist_ok=True)

    measurement_names = ["agent_data", "simulation_params", "resource_data"]
    for mes_name in measurement_names:
        data_dict = ifclient.query(f"select * from {mes_name}", chunked=True, chunk_size=100000)
        ret = data_dict[mes_name]
        save_file_path = os.path.join(save_dir, f'{mes_name}.csv')
        ret.to_csv(save_file_path, sep=",", encoding="utf-8")
