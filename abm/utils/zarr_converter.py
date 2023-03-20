import os
import uuid
from pathlib import Path

import numpy as np

from abm.loader.data_loader import ExperimentLoader
import pandas as pd


def zarr_to_csv_coop_signaling_results(data_folder):
    """Converts a zarr file to a csv file.

    """
    experiment = ExperimentLoader(data_folder)

    # data shape is (batch, simulation parameter 1 ... simulation parameter n, agent number, timestep)
    pos_x = experiment.agent_summary['posx']
    pos_y = experiment.agent_summary['posy']
    orientation = experiment.agent_summary['orientation']
    # agent_modes = experiment.agent_summary['mode']
    res_pos_x = experiment.res_summary['posx']
    res_pos_y = experiment.res_summary['posy']
    # meter = experiment.agent_summary['meter']
    sig = experiment.agent_summary['signalling']
    score = experiment.agent_summary['collresource']
    num_batches = experiment.num_batches

    # convert zarr to the tabular format with columns: time, arent ID/resource, x, y, orientation, signaling, score
    # NOTE data are stored in the long format, so each row is a single agent/resource at a single timestep

    for i, s in enumerate(experiment.varying_params['RES_VEL']):
        for j, v in enumerate(experiment.varying_params['SIGNALLING_PROB']):
            folder_name = f"signalling_{s}_vel_{v}"
            # create a folder with the condition name (parameter values) using pathlib
            condition_folder = Path(data_folder) / folder_name
            condition_folder.mkdir(parents=True, exist_ok=True)
            time = np.arange(0, experiment.chunksize)
            for batch in range(num_batches):
                data = []
                ids = ["resource"]
                data.append(pd.DataFrame({
                    'Time_ms': time,
                    'id': [0] * len(time),
                    'positions_x': res_pos_x[batch, i, j, 0, :],
                    'positions_z': res_pos_y[batch, i, j, 0, :],
                    'rotation_y': orientation[batch, i, j, 0, :],
                    'is_signaling': [0] * len(time),
                    'score': [0] * len(time)
                }))

                for a in range(pos_x.shape[-2]):
                    agent_id = uuid.uuid4().hex
                    data.append(pd.DataFrame({
                        'Time_ms': time,
                        'id': [a + 1] * len(time),
                        'positions_x': pos_x[batch, i, j, a, :],
                        'positions_z': pos_y[batch, i, j, a, :],
                        'orientation': orientation[batch, i, j, a, :],
                        'is_signaling': sig[batch, i, j, a, :],
                        'score': score[batch, i, j, a, :]
                    }))
                    ids.append(agent_id)
                data = pd.concat(data)
                # sort by time and agent ID
                data = data.sort_values(by=['time', 'agent ID'], ignore_index=True)
                data.to_csv(os.path.join(condition_folder, f"{'_'.join(ids)}.csv"), index=False)


if __name__ == "__main__":
    path_to_simulation_data = ""
    zarr_to_csv_coop_signaling_results(path_to_simulation_data)
