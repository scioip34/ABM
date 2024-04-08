"""Script to export zarr format files to HD5 format so that analytic
group can also access the data from outside python"""

import h5py
import zarr
import os
import glob

exppath = "/home/mezey/Desktop/Database/simulation_data/humanexp1"
undersample = 1
usefirst_n_batch = 20
zarr_format = ".zarr"

summarydir = os.path.join(exppath, "summary")
arrays_to_export = glob.glob(os.path.join(summarydir, f"*{zarr_format}"))
hd5dir = os.path.join(summarydir, "HDF5_DATA")
if not os.path.isdir(hd5dir):
    os.makedirs(hd5dir, exist_ok=True)
h5f_ag = h5py.File(os.path.join(hd5dir, "agent_data.h5"), 'r')
h5f_res = h5py.File(os.path.join(hd5dir, "resource_data.h5"), 'r')

for array_path in arrays_to_export:
    array_name = os.path.basename(array_path)
    print(f"Exporting zarr array: {array_name}")
    sourcearr = zarr.open(os.path.join(summarydir, array_name), mode="r")
    numbatches = sourcearr.shape[0]
    for batch_i in range(numbatches):
        if batch_i < usefirst_n_batch:
            print(f"Exporting batch {batch_i}")
            if array_name.startswith("agent"):
                h5f_ag.create_dataset(f'{array_name.split(".")[0]}_{batch_i}', data=sourcearr[batch_i, ..., ::undersample])
            else:
                h5f_res.create_dataset(f'{array_name.split(".")[0]}_{batch_i}', data=sourcearr[batch_i, ..., ::undersample])
        else:
            print(f"Ignoring batch {batch_i} as requested!")

h5f_ag.close()
h5f_res.close()