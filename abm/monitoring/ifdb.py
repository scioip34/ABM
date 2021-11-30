"""
@author: mezdahun
@description: Helper functions for InfluxDB
"""
from influxdb import InfluxDBClient
import abm.contrib.ifdb_params as ifdbp
import datetime


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


