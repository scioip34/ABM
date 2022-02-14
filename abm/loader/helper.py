"""
helper.py : including helper functions for loading and handling data
"""
import csv
import numpy as np
from itertools import islice


def load_csv_file(path, undersample=100):
    """Loading csv files into memory (dictionaries)."""
    with open(path, mode='r') as infile:
        reader = csv.reader(infile)
        new_dict = {}
        dict_keys = []
        main_iter = iter(enumerate(reader))
        # for ri, row in enumerate(reader):
        #     if ri == 0:
        #         dict_keys = list(row)
        #         dict_keys = ["t" if x == "" else x for x in dict_keys]
        #         for key in dict_keys:
        #             new_dict[key] = []
        #     else:
        #         if ri % undersample == 0:
        #             # print("ADD ROW", ri)
        #             for ci, key in enumerate(dict_keys):
        #                 new_dict[key].append(row[ci])
        for ri, row in main_iter:
            if ri == 0:
                dict_keys = list(row)
                dict_keys = ["t" if x == "" else x for x in dict_keys]
                for key in dict_keys:
                    new_dict[key] = []
            else:
                for ci, key in enumerate(dict_keys):
                    new_dict[key].append(row[ci])
                [next(main_iter, None) for _ in range(undersample)]  # skip 5
        return new_dict


def reconstruct_VPF(VPF_resolution, up_edge_list, down_edge_list):
    """Constructing a binary VPF array (1/0) according to saved edge data.

    :param VPF_resolution: length of VPF array
    :param up_edge_list: list of indices where VPF have uprising edge (from 0 to 1)
    :param down_edge_list: list of indices where VPF has downfalling edge (1 to 0)"""
    if len(up_edge_list) != len(down_edge_list):
        raise Exception("Length of up and down edges differ during VPF reconstruction.")

    VPF = np.zeros(VPF_resolution)
    for i in range(len(up_edge_list)):
        VPF[up_edge_list[i]:down_edge_list[i]] = 1
    return VPF

