"""
@author: mezdahun
@description: Helper functions for InfluxDB
"""
import datetime
import importlib
import os
import sys

import numpy as np
from influxdb import InfluxDBClient, DataFrameClient

import abm.contrib.ifdb_params as ifdbp
from abm.loader.helper import reconstruct_VPF

batch_bodies_agents = []
batch_bodies_resources = []
resources_dict = {}
agents_dict = {}

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
                              ifdbp.INFLUX_DB_NAME,
                              timeout=ifdbp.INFLUX_TIMEOUT,
                              retries=ifdbp.INFLUX_RETRIES)
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

def save_agent_data_RAM(agents, t):
    """Saving relevant agent data into InfluxDB intance
    if multiple simulations are running in parallel a uuid hash must be passed as experiment hash to find
    the unique measurement in the database
    """
    global agents_dict
    if t % 500 == 0:
        print(f"Agent data size in memory: {sys.getsizeof(agents_dict)/1024} MB", )
    for agent in agents:
        if agent.id not in list(agents_dict.keys()):
            agents_dict[agent.id] = {}
            agent_name = f"agent-{pad_to_n_digits(agent.id, n=2)}"
            agents_dict[agent.id]['agent_name'] = agent_name
            agents_dict[agent.id][f"posx"] = []
            agents_dict[agent.id][f"posy"] = []
            agents_dict[agent.id][f"orientation"] = []
            agents_dict[agent.id][f"velocity"] = []
            agents_dict[agent.id][f"w"] = []
            agents_dict[agent.id][f"u"] = []
            agents_dict[agent.id][f"Ipriv"] = []
            agents_dict[agent.id][f"mode"] = []
            agents_dict[agent.id][f"collectedr"] = []
            agents_dict[agent.id][f"expl_patch_id"] = []
            # only storing visual field edges to compress data and keep real time simulations
            agents_dict[agent.id][f"vfield_up"] = []
            agents_dict[agent.id][f"vfield_down"] = []

        # format the data as a single measurement for influx
        agents_dict[agent.id][f"posx"].append(int(agent.position[0]))
        agents_dict[agent.id][f"posy"].append(int(agent.position[1]))
        agents_dict[agent.id][f"orientation"].append(float(agent.orientation))
        agents_dict[agent.id][f"velocity"].append(float(agent.velocity))
        agents_dict[agent.id][f"w"].append(float(agent.w))
        agents_dict[agent.id][f"u"].append(float(agent.u))
        agents_dict[agent.id][f"Ipriv"].append(float(agent.I_priv))
        agents_dict[agent.id][f"mode"].append(int(mode_to_int(agent.mode)))
        agents_dict[agent.id][f"collectedr"].append(float(agent.collected_r))
        agents_dict[agent.id][f"expl_patch_id"].append(int(agent.exploited_patch_id))
        # only storing visual field edges to compress data and keep real time simulations
        agents_dict[agent.id][f"vfield_up"].append(f"{np.where(np.roll(agent.soc_v_field,1) < agent.soc_v_field)[0]}")
        agents_dict[agent.id][f"vfield_down"].append(f"{np.where(np.roll(agent.soc_v_field, 1) > agent.soc_v_field)[0]}")


def save_agent_data(ifclient, agents, t, exp_hash="", batch_size=None):
    """Saving relevant agent data into InfluxDB intance
    if multiple simulations are running in parallel a uuid hash must be passed as experiment hash to find
    the unique measurement in the database
    """
    global batch_bodies_agents
    measurement_name = f"agent_data{exp_hash}"
    fields = {}
    fields['t'] = t
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

    body = {
            "measurement": measurement_name,
            "time": time,
            "fields": fields
        }

    # write the measurement in batches
    batch_bodies_agents.append(body)
    if batch_size is None:
        batch_size = ifdbp.write_batch_size
        # todo: in the last timestep we need to try until we can
    if len(batch_bodies_agents) % batch_size == 0 and len(batch_bodies_agents) != 0:
        try:
            ifclient.write_points(batch_bodies_agents)
            batch_bodies_agents = []
        except Exception as e:
            print(f"Could not write in database, got Influx Error, will try to push with next batch...")
            print(e)
    # todo: in the last timestep use this code
    # if len(batch_bodies_agents) == batch_size:
    #     write_success = False
    #     retries = 0
    #     while not write_success and retries < 100:
    #         try:
    #             retries += 1
    #             ifclient.write_points(batch_bodies_agents)
    #             write_success = True
    #         except InfluxDBServerError as e:
    #             print(f"INFLUX ERROR, will retry {retries}")
    #             print(e)
    #     if retries == 100:
    #         raise Exception("Too many retries to write to InfluxDB instance. Stopping application!")
    #     batch_bodies_agents = []


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


def save_resource_data_RAM(resources, t):
    """Saving relevant resource patch data into InfluxDB instance
    if multiple simulations are running in parallel a uuid hash must be passed as experiment hash to find
    the unique measurement in the database"""
    global resources_dict
    if t % 500 == 0:
        print(f"Resource data size in memory: {sys.getsizeof(resources_dict)/1024} MB", )
    ids_in_run = []
    for res in resources:
        if res.id not in list(resources_dict.keys()):
            resources_dict[res.id] = {}
            resources_dict[res.id]["start_time"] = t
            resources_dict[res.id]["end_time"] = None

            res_name = f"res-{pad_to_n_digits(res.id, n=3)}"
            resources_dict[res.id]["res_name"] = res_name

            # format the data as a single measurement for influx
            # pos and radius are enough to calculate center
            # (only important in spatially moving res, otherwise take the first element)
            # wasteful with resources but generalizable for later with no effort
            resources_dict[res.id]["pos_x"] = []
            resources_dict[res.id]["pos_y"] = []
            resources_dict[res.id]["radius"] = int(res.radius)
            resources_dict[res.id]["resc_left"] = []
            resources_dict[res.id]["quality"] = []

        resources_dict[res.id]["pos_x"].append(int(res.position[0]))
        resources_dict[res.id]["pos_y"].append(int(res.position[1]))
        resources_dict[res.id]["resc_left"].append(float(res.resc_left))
        resources_dict[res.id]["quality"].append(float(res.unit_per_timestep))
        # ids_in_run.append(res.id)
    # global_ids = set(list(resources_dict.keys()))
    # disappeared_patch_ids = list(global_ids.difference(set(ids_in_run)))
    # for res_id in disappeared_patch_ids:
    #     if resources_dict[res_id]["end_time"] is None:
    #         resources_dict[res_id]["end_time"] = t


def save_resource_data(ifclient, resources, t, exp_hash="", batch_size=None):
    """Saving relevant resource patch data into InfluxDB instance
    if multiple simulations are running in parallel a uuid hash must be passed as experiment hash to find
    the unique measurement in the database"""
    global batch_bodies_resources
    measurement_name = f"resource_data{exp_hash}"
    fields = {}
    fields['t'] = t
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

    body = {
            "measurement": measurement_name,
            "time": time,
            "fields": fields
        }

    batch_bodies_resources.append(body)
    # write the measurement in batches
    if batch_size is None:
        batch_size = ifdbp.write_batch_size
    # todo: in the last timestep we need to try until we can
    if len(batch_bodies_resources) % batch_size == 0 and len(batch_bodies_resources) != 0:
        try:
            ifclient.write_points(batch_bodies_resources)
            batch_bodies_resources = []
        except Exception as e:
            print(f"Could not write in database, got Influx Error, will try to push with next batch...")
            print(e)


def save_simulation_params(ifclient, sim, exp_hash=""):
    """saving simulation parameters to IFDB
    if multiple simulations are running in parallel a uuid hash must be passed as experiment hash to find
    the unique measurement in the database"""

    measurement_name = f"simulation_params{exp_hash}"
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


def save_ifdb_as_csv(exp_hash="", use_ram=False, as_zar=True, save_extracted_vfield=False):
    """Saving the whole influx database as a single csv file
    if multiple simulations are running in parallel a uuid hash must be passed as experiment hash to find
    the unique measurement in the database"""
    global resources_dict, agents_dict
    importlib.reload(ifdbp)
    if not use_ram:
        print("Saving data with client timeout of 600s")
        ifclient = DataFrameClient(ifdbp.INFLUX_HOST,
                                   ifdbp.INFLUX_PORT,
                                   ifdbp.INFLUX_USER,
                                   ifdbp.INFLUX_PSWD,
                                   ifdbp.INFLUX_DB_NAME,
                                   timeout=600,  # using larger timeout for data saving
                                   retries=ifdbp.INFLUX_RETRIES)

        # create base folder in data
        save_dir = ifdbp.TIMESTAMP_SAVE_DIR
        os.makedirs(save_dir, exist_ok=True)

        measurement_names = [f"agent_data{exp_hash}", f"resource_data{exp_hash}"]
        for mes_name in measurement_names:
            print(f"Querying data with measurement name: {mes_name}")
            data_dict = ifclient.query(f"select * from {mes_name}", chunked=True, chunk_size=110000)
            print(f"Queried data size in memory: {sys.getsizeof(data_dict)/1024} MB", )
            print("Keys: ", list(data_dict.keys()))
            ret = data_dict[mes_name]
            if exp_hash != "":
                filename = mes_name.split(exp_hash)[0]
            else:
                filename = mes_name
            print("Saving csv file...")
            save_file_path = os.path.join(save_dir, f'{filename}.csv')
            ret.to_csv(save_file_path, sep=",", encoding="utf-8")
            print(f"Cleaning up measurement from IFDB: {mes_name}")
            ifclient.delete_series(ifdbp.INFLUX_DB_NAME, mes_name)
            print(f"Remaining measurements: {ifclient.get_list_measurements()}")

    else:
        save_dir = ifdbp.TIMESTAMP_SAVE_DIR
        os.makedirs(save_dir, exist_ok=True)
        if not as_zar:
            import json
            print("Saving resource data as json file...")
            mes_name = f"resource_data{exp_hash}"
            if exp_hash != "":
                filename = mes_name.split(exp_hash)[0]
            else:
                filename = mes_name
            save_file_path = os.path.join(save_dir, f'{filename}.json')
            with open(save_file_path, "w") as f:
                json.dump(resources_dict, f,)

            print("Saving agent data as json file...")
            mes_name = f"agent_data{exp_hash}"
            if exp_hash != "":
                filename = mes_name.split(exp_hash)[0]
            else:
                filename = mes_name
            save_file_path = os.path.join(save_dir, f'{filename}.json')
            with open(save_file_path, "w") as f:
                json.dump(agents_dict, f)
        else:
            import zarr, json
            print("Saving resource data as compressed zarr arrays...")
            num_res = len(resources_dict)
            t_len = len(resources_dict[list(resources_dict.keys())[0]]['pos_x'])
            posxzarr = zarr.open(os.path.join(save_dir, "res_posx.zarr"), mode='w', shape=(num_res, t_len),
                                 chunks = (num_res, t_len), dtype = 'float')
            posyzarr = zarr.open(os.path.join(save_dir, "res_posy.zarr"), mode='w', shape=(num_res, t_len),
                                 chunks = (num_res, t_len), dtype = 'float')
            rleftzarr = zarr.open(os.path.join(save_dir, "res_left.zarr"), mode='w', shape=(num_res, t_len),
                                  chunks = (num_res, t_len), dtype = 'float')
            qualzarr = zarr.open(os.path.join(save_dir, "res_qual.zarr"), mode='w', shape=(num_res, t_len),
                                 chunks = (num_res, t_len), dtype = 'float')
            resrad = zarr.open(os.path.join(save_dir, "res_rad.zarr"), mode='w', shape=(num_res, t_len),
                               chunks = (num_res, t_len), dtype = 'float')
            for res_id, res_dict in resources_dict.items():
                posxzarr[res_id-1, :] = resources_dict[res_id]['pos_x']
                posyzarr[res_id-1, :] = resources_dict[res_id]['pos_y']
                rleftzarr[res_id-1, :] = resources_dict[res_id]['resc_left']
                qualzarr[res_id-1, :] = resources_dict[res_id]['quality']
                resrad[res_id-1, :] = [resources_dict[res_id]['radius'] for i in range(t_len)]

            print("Saving agent data as compressed zarr arrays...")
            num_ag = len(agents_dict)
            v_field_len = int(float(ifdbp.envconf.get("VISUAL_FIELD_RESOLUTION")))
            aposxzarr = zarr.open(os.path.join(save_dir, "ag_posx.zarr"), mode='w', shape=(num_ag, t_len),
                                 chunks=(num_ag, t_len), dtype='float')
            aposyzarr = zarr.open(os.path.join(save_dir, "ag_posy.zarr"), mode='w', shape=(num_ag, t_len),
                                 chunks=(num_ag, t_len), dtype='float')
            aorizarr = zarr.open(os.path.join(save_dir, "ag_ori.zarr"), mode='w', shape=(num_ag, t_len),
                                 chunks=(num_ag, t_len), dtype='float')
            avelzarr = zarr.open(os.path.join(save_dir, "ag_vel.zarr"), mode='w', shape=(num_ag, t_len),
                                 chunks=(num_ag, t_len), dtype='float')
            awzarr = zarr.open(os.path.join(save_dir, "ag_w.zarr"), mode='w', shape=(num_ag, t_len),
                                 chunks=(num_ag, t_len), dtype='float')
            auzarr = zarr.open(os.path.join(save_dir, "ag_u.zarr"), mode='w', shape=(num_ag, t_len),
                                 chunks=(num_ag, t_len), dtype='float')
            aiprivzarr = zarr.open(os.path.join(save_dir, "ag_ipriv.zarr"), mode='w', shape=(num_ag, t_len),
                                 chunks=(num_ag, t_len), dtype='float')
            amodezarr = zarr.open(os.path.join(save_dir, "ag_mode.zarr"), mode='w', shape=(num_ag, t_len),
                                 chunks=(num_ag, t_len), dtype='float')
            acollrzarr = zarr.open(os.path.join(save_dir, "ag_collr.zarr"), mode='w', shape=(num_ag, t_len),
                                  chunks=(num_ag, t_len), dtype='float')
            aexplrzarr = zarr.open(os.path.join(save_dir, "ag_explr.zarr"), mode='w', shape=(num_ag, t_len),
                                  chunks=(num_ag, t_len), dtype='float')
            if v_field_len is not None and save_extracted_vfield:
                avfzarr = zarr.open(os.path.join(save_dir, "ag_vf.zarr"), mode='w', shape=(num_ag, t_len, v_field_len),
                                      chunks=(num_ag, 1, v_field_len), dtype='float')

            for ag_id, ag_ict in agents_dict.items():
                aposxzarr[ag_id-1, :] = agents_dict[ag_id]['posx']
                aposyzarr[ag_id-1, :] = agents_dict[ag_id]['posy']
                aorizarr[ag_id-1, :] = agents_dict[ag_id]['orientation']
                avelzarr[ag_id-1, :] = agents_dict[ag_id]['velocity']
                awzarr[ag_id-1, :] = agents_dict[ag_id]['w']
                auzarr[ag_id-1, :] = agents_dict[ag_id]['u']
                aiprivzarr[ag_id-1, :] = agents_dict[ag_id]['Ipriv']
                amodezarr[ag_id-1, :] = agents_dict[ag_id]['mode']
                acollrzarr[ag_id-1, :] = agents_dict[ag_id]['collectedr']
                aexplrzarr[ag_id-1, :] = agents_dict[ag_id]['expl_patch_id']
                if v_field_len is not None and save_extracted_vfield:
                    agents_dict[ag_id]['vfield_up'] = np.array(
                        [i.replace("   ", " ").replace("  ", " ").replace("[  ", "[").replace(
                            "[ ", "[").replace(" ", ", ") for i in agents_dict[ag_id]['vfield_up']], dtype=object)
                    agents_dict[ag_id]['vfield_down'] = np.array(
                        [i.replace("   ", " ").replace("  ", " ").replace("[  ", "[").replace(
                            "[ ", "[").replace(" ", ", ") for i in agents_dict[ag_id]['vfield_down']], dtype=object)
                    for t in range(t_len):
                        vfup = json.loads(agents_dict[ag_id]['vfield_up'][t])
                        vfdown = json.loads(agents_dict[ag_id]['vfield_down'][t])
                        if vfup != []:
                            vfup = [int(float(v)) for v in vfup]
                            vfdown = [int(float(v)) for v in vfdown]
                            vf = reconstruct_VPF(v_field_len, vfup, vfdown)
                        else:
                            vf = np.zeros(v_field_len)
                        avfzarr[ag_id-1, t, :] = vf

        print("Cleaning global data structure!")
        resources_dict = {}
        agents_dict = {}

